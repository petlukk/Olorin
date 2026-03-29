// Reference implementation: Llama 3 pre-tokenizer
// Produces span boundaries matching the tiktoken regex:
//   (?:'[sStTrReEvVmMlLdD]{1,2})|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Input:  text (raw bytes), len
// Output: boundaries[i] = 1 if position i starts a new span, else 0
//
// Flag encoding (from byte_classifier):
//   1 = whitespace, 2 = letter, 4 = digit, 8 = punctuation, 16 = non-ASCII

#include <stdint.h>
#include <string.h>

static uint8_t classify(uint8_t b) {
    if (b == 32 || b == 9 || b == 10 || b == 13) return 1;
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) return 2;
    if (b >= '0' && b <= '9') return 4;
    if (b > 127) return 16;
    if (b >= 33 && b <= 126) return 8;
    return 0;
}

// Check if byte is a contraction suffix start after apostrophe
// Matches: 's 't 're 've 'm 'll 'd (case-insensitive)
static int is_contraction_suffix(const uint8_t *text, int pos, int len) {
    if (pos >= len) return 0;
    uint8_t c = text[pos] | 32; // lowercase
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 1;
    if (pos + 1 < len) {
        uint8_t c2 = text[pos + 1] | 32;
        if (c == 'r' && c2 == 'e') return 2;
        if (c == 'v' && c2 == 'e') return 2;
        if (c == 'l' && c2 == 'l') return 2;
    }
    return 0;
}

void pretokenize_ref(const uint8_t *text, uint8_t *boundaries, int len) {
    if (len <= 0) return;
    memset(boundaries, 0, len);

    int i = 0;
    while (i < len) {
        boundaries[i] = 1; // every span starts with boundary=1

        uint8_t f = classify(text[i]);

        // Contraction: '[sStTdDmM] or '[rReEvVlL][eElL]
        if (text[i] == '\'' && i + 1 < len) {
            int suffix_len = is_contraction_suffix(text, i + 1, len);
            if (suffix_len > 0) {
                i += 1 + suffix_len;
                continue;
            }
        }

        // Letters (possibly preceded by one non-letter/non-digit/non-newline)
        if (f == 2 || f == 16) {
            i++;
            while (i < len) {
                uint8_t nf = classify(text[i]);
                if (nf != 2 && nf != 16) break;
                i++;
            }
            continue;
        }

        // Digits: groups of 1-3
        if (f == 4) {
            int count = 0;
            while (i < len && classify(text[i]) == 4 && count < 3) {
                i++;
                count++;
            }
            continue;
        }

        // Whitespace that includes newlines: \s*[\r\n]+
        if (f == 1 && (text[i] == 10 || text[i] == 13)) {
            while (i < len && classify(text[i]) == 1) i++;
            continue;
        }

        // Regular whitespace: \s+
        if (f == 1) {
            while (i < len && classify(text[i]) == 1) i++;
            continue;
        }

        // Punctuation / other: [^\s\p{L}\p{N}]+[\r\n]*
        if (f == 8) {
            while (i < len && classify(text[i]) == 8) i++;
            // consume trailing newlines
            while (i < len && (text[i] == 10 || text[i] == 13)) i++;
            continue;
        }

        // Fallback: single byte
        i++;
    }
}
