#include <capnp/serialize.h>
#include <zstd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

namespace {

constexpr std::array<char, 8> kMagic = {
  'B', 'L', 'A', 'T', 'V', '2', 'R', '1',
};
constexpr uint32_t kStreamSchemaVersion = 1;
constexpr uint32_t kEndRecord = std::numeric_limits<uint32_t>::max();
constexpr size_t kInputChunkSize = 256 * 1024;
constexpr size_t kMaximumSegmentCount = 512;
constexpr size_t kMaximumMessageBytes = 64 * 1024 * 1024;
constexpr int kMaximumZstdWindowLog = 27;  // 128 MiB hard decoder bound.

bool writeU32(uint32_t value) {
  const std::array<char, 4> encoded = {
    static_cast<char>(value & 0xffU),
    static_cast<char>((value >> 8U) & 0xffU),
    static_cast<char>((value >> 16U) & 0xffU),
    static_cast<char>((value >> 24U) & 0xffU),
  };
  std::cout.write(encoded.data(), encoded.size());
  return std::cout.good();
}

bool writeU64(uint64_t value) {
  const std::array<char, 8> encoded = {
    static_cast<char>(value & 0xffULL),
    static_cast<char>((value >> 8U) & 0xffULL),
    static_cast<char>((value >> 16U) & 0xffULL),
    static_cast<char>((value >> 24U) & 0xffULL),
    static_cast<char>((value >> 32U) & 0xffULL),
    static_cast<char>((value >> 40U) & 0xffULL),
    static_cast<char>((value >> 48U) & 0xffULL),
    static_cast<char>((value >> 56U) & 0xffULL),
  };
  std::cout.write(encoded.data(), encoded.size());
  return std::cout.good();
}

uint32_t readU32(const uint8_t *data) {
  return (
    static_cast<uint32_t>(data[0])
    | (static_cast<uint32_t>(data[1]) << 8U)
    | (static_cast<uint32_t>(data[2]) << 16U)
    | (static_cast<uint32_t>(data[3]) << 24U)
  );
}

bool selected(cereal::Event::Which which) {
  switch (which) {
    case cereal::Event::INIT_DATA:
    case cereal::Event::SENTINEL:
    case cereal::Event::CONTROLS_STATE:
    case cereal::Event::CAR_STATE:
    case cereal::Event::CAR_CONTROL:
    case cereal::Event::CAR_OUTPUT:
    case cereal::Event::LIVE_PARAMETERS:
    case cereal::Event::CAR_PARAMS:
      return true;
    default:
      return false;
  }
}

class MessageStream {
public:
  bool append(const uint8_t *data, size_t size) {
    if (size == 0) {
      return true;
    }
    if (offset_ > 0 && (
      offset_ >= 4 * 1024 * 1024
      || offset_ * 2 >= pending_.size()
    )) {
      pending_.erase(
        pending_.begin(),
        pending_.begin() + static_cast<std::ptrdiff_t>(offset_)
      );
      offset_ = 0;
    }
    pending_.insert(pending_.end(), data, data + size);
    return drain(false);
  }

  bool finish() {
    if (!drain(true)) {
      return false;
    }
    if (offset_ != pending_.size()) {
      std::cerr << "truncated capnp message at end of rlog\n";
      return false;
    }
    return true;
  }

  uint64_t emittedCount() const {
    return emitted_count_;
  }

private:
  bool nextMessageSize(size_t *message_size) const {
    const size_t available = pending_.size() - offset_;
    if (available < 8) {
      return false;
    }
    const uint8_t *message = pending_.data() + offset_;
    const uint64_t segment_count =
      static_cast<uint64_t>(readU32(message)) + 1ULL;
    if (segment_count == 0 || segment_count > kMaximumSegmentCount) {
      throw std::runtime_error("invalid capnp segment count");
    }
    const uint64_t table_u32 = 1ULL + segment_count;
    const uint64_t padded_table_u32 = (table_u32 + 1ULL) & ~1ULL;
    const uint64_t table_bytes = padded_table_u32 * sizeof(uint32_t);
    if (table_bytes > available) {
      return false;
    }
    uint64_t content_words = 0;
    for (uint64_t i = 0; i < segment_count; ++i) {
      content_words += readU32(
        message + sizeof(uint32_t) * (1ULL + i)
      );
      if (
        content_words
        > (kMaximumMessageBytes - table_bytes) / sizeof(capnp::word)
      ) {
        throw std::runtime_error("capnp message exceeds size bound");
      }
    }
    const uint64_t total =
      table_bytes + content_words * sizeof(capnp::word);
    if (total == 0 || total > kMaximumMessageBytes) {
      throw std::runtime_error("invalid capnp message size");
    }
    if (total > available) {
      return false;
    }
    *message_size = static_cast<size_t>(total);
    return true;
  }

  bool drain(bool final) {
    try {
      while (offset_ < pending_.size()) {
        size_t message_size = 0;
        if (!nextMessageSize(&message_size)) {
          if (final) {
            return false;
          }
          break;
        }
        const uint8_t *message = pending_.data() + offset_;
        // vector<uint8_t> does not promise capnp::word alignment. Copy one
        // bounded message into explicitly word-aligned storage before asking
        // Cap'n Proto to decode it; stdout still receives the original bytes.
        std::vector<capnp::word> aligned(
          message_size / sizeof(capnp::word)
        );
        std::memcpy(aligned.data(), message, message_size);
        capnp::FlatArrayMessageReader reader(kj::arrayPtr(
          aligned.data(),
          aligned.size()
        ));
        const auto event = reader.getRoot<cereal::Event>();
        const auto which = event.which();
        if (terminal_seen_) {
          throw std::runtime_error(
            "event appears after terminal rlog sentinel"
          );
        }
        if (event.getValid() && which == cereal::Event::SENTINEL) {
          const auto sentinel = event.getSentinel().getType();
          terminal_seen_ = (
            sentinel == cereal::Sentinel::SentinelType::END_OF_SEGMENT
            || sentinel == cereal::Sentinel::SentinelType::END_OF_ROUTE
          );
        }
        if (selected(which)) {
          if (
            !writeU32(static_cast<uint32_t>(message_size))
            || !writeU32(static_cast<uint32_t>(which))
            || !writeU64(event.getLogMonoTime())
          ) {
            std::cerr << "could not write extraction stream\n";
            return false;
          }
          std::cout.write(
            reinterpret_cast<const char *>(message),
            static_cast<std::streamsize>(message_size)
          );
          if (!std::cout.good()) {
            std::cerr << "could not write extraction payload\n";
            return false;
          }
          ++emitted_count_;
        }
        offset_ += message_size;
      }
    } catch (const kj::Exception &error) {
      std::cerr << "capnp parse failed: "
                << error.getDescription().cStr() << "\n";
      return false;
    } catch (const std::exception &error) {
      std::cerr << error.what() << "\n";
      return false;
    }
    return true;
  }

  std::vector<uint8_t> pending_;
  size_t offset_ = 0;
  uint64_t emitted_count_ = 0;
  bool terminal_seen_ = false;
};

bool extractRaw(std::ifstream &input, MessageStream *messages) {
  std::array<uint8_t, kInputChunkSize> chunk = {};
  while (input.good()) {
    input.read(
      reinterpret_cast<char *>(chunk.data()),
      static_cast<std::streamsize>(chunk.size())
    );
    const auto count = input.gcount();
    if (count > 0 && !messages->append(
      chunk.data(),
      static_cast<size_t>(count)
    )) {
      return false;
    }
  }
  return input.eof() && messages->finish();
}

bool extractZstd(std::ifstream &input, MessageStream *messages) {
  ZSTD_DStream *stream = ZSTD_createDStream();
  if (stream == nullptr) {
    std::cerr << "could not create zstd stream\n";
    return false;
  }
  const size_t initialized = ZSTD_initDStream(stream);
  if (ZSTD_isError(initialized)) {
    std::cerr << ZSTD_getErrorName(initialized) << "\n";
    ZSTD_freeDStream(stream);
    return false;
  }
  const size_t window_limit = ZSTD_DCtx_setParameter(
    stream,
    ZSTD_d_windowLogMax,
    kMaximumZstdWindowLog
  );
  if (ZSTD_isError(window_limit)) {
    std::cerr << ZSTD_getErrorName(window_limit) << "\n";
    ZSTD_freeDStream(stream);
    return false;
  }

  std::vector<uint8_t> compressed(ZSTD_DStreamInSize());
  std::vector<uint8_t> decompressed(ZSTD_DStreamOutSize());
  size_t remaining = 1;
  bool frame_complete = false;
  bool success = true;
  while (input.good()) {
    input.read(
      reinterpret_cast<char *>(compressed.data()),
      static_cast<std::streamsize>(compressed.size())
    );
    const size_t count = static_cast<size_t>(input.gcount());
    if (count == 0) {
      break;
    }
    if (frame_complete) {
      std::cerr << "trailing data after zstd frame\n";
      success = false;
      break;
    }
    ZSTD_inBuffer in = {compressed.data(), count, 0};
    bool output_was_full = false;
    do {
      ZSTD_outBuffer out = {
        decompressed.data(),
        decompressed.size(),
        0,
      };
      remaining = ZSTD_decompressStream(stream, &out, &in);
      if (ZSTD_isError(remaining)) {
        std::cerr << ZSTD_getErrorName(remaining) << "\n";
        success = false;
        break;
      }
      if (!messages->append(decompressed.data(), out.pos)) {
        success = false;
        break;
      }
      if (remaining == 0) {
        frame_complete = true;
        if (in.pos != in.size) {
          std::cerr << "trailing data after zstd frame\n";
          success = false;
        }
        break;
      }
      output_was_full = out.pos == out.size;
      // Keep draining when zstd filled the output exactly, even if it also
      // consumed the final input byte. This avoids a false truncation at an
      // output-buffer boundary. A non-full output with no input left needs
      // the next compressed chunk instead.
    } while (
      in.pos < in.size
      || (output_was_full && remaining > 0)
    );
    if (!success) {
      break;
    }
  }
  if (!input.eof() || !frame_complete || remaining != 0) {
    std::cerr << "truncated zstd frame\n";
    success = false;
  }
  if (success) {
    success = messages->finish();
  }
  ZSTD_freeDStream(stream);
  return success;
}

bool isZstd(std::ifstream &input) {
  std::array<uint8_t, 4> magic = {};
  input.read(
    reinterpret_cast<char *>(magic.data()),
    static_cast<std::streamsize>(magic.size())
  );
  const bool zstd = (
    input.gcount() == static_cast<std::streamsize>(magic.size())
    && magic[0] == 0x28
    && magic[1] == 0xb5
    && magic[2] == 0x2f
    && magic[3] == 0xfd
  );
  input.clear();
  input.seekg(0);
  return zstd;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: blatv2_rlog_extract RLOG_OR_RLOG_ZST\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input.is_open()) {
    std::cerr << "could not open rlog\n";
    return 3;
  }

  std::cout.write(kMagic.data(), kMagic.size());
  if (!writeU32(kStreamSchemaVersion) || !writeU32(0)) {
    return 4;
  }
  MessageStream messages;
  const bool success = (
    isZstd(input)
    ? extractZstd(input, &messages)
    : extractRaw(input, &messages)
  );
  if (!success) {
    return 5;
  }
  if (
    !writeU32(0)
    || !writeU32(kEndRecord)
    || !writeU64(messages.emittedCount())
  ) {
    return 6;
  }
  std::cout.flush();
  return std::cout.good() ? 0 : 7;
}
