/// Splits `text` on blank lines and regroups paragraphs into chunks up to
/// `max_chunk_chars`, so a large document becomes several independently-
/// searchable/observable pieces instead of one large one. A single paragraph
/// longer than `max_chunk_chars` (some PDF extractions produce long runs of
/// text with no blank-line breaks at all) is itself regrouped on whitespace,
/// only cutting mid-word - at a UTF-8 char boundary - for a lone "word" still
/// too long by itself: no chunk this returns can ever exceed
/// `max_chunk_chars`, which matters because downstream embedding models
/// reject inputs past their context window. Empty/whitespace-only input
/// yields no chunks. Shared by anything that ingests external text
/// (Knowledge Library URL ingestion, the household PDF library reader) so
/// chunking behavior stays identical across ingestion paths.
pub fn chunk_text(text: &str, max_chunk_chars: usize) -> Vec<String> {
    let paragraphs = text.split("\n\n").map(str::trim).filter(|p| !p.is_empty());
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in paragraphs {
        if paragraph.len() > max_chunk_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            split_oversized_paragraph(paragraph, max_chunk_chars, &mut chunks);
            continue;
        }
        if !current.is_empty() && current.len() + paragraph.len() + 2 > max_chunk_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Regroups a single paragraph that's already too big for `max_chunk_chars`
/// on its own, word by word, so its pieces still respect the same limit.
fn split_oversized_paragraph(paragraph: &str, max_chunk_chars: usize, chunks: &mut Vec<String>) {
    let mut current = String::new();
    for word in paragraph.split_whitespace() {
        if word.len() > max_chunk_chars {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            let mut rest = word;
            while rest.len() > max_chunk_chars {
                let mut split_at = max_chunk_chars;
                while !rest.is_char_boundary(split_at) {
                    split_at -= 1;
                }
                let (piece, remainder) = rest.split_at(split_at);
                chunks.push(piece.to_string());
                rest = remainder;
            }
            if !rest.is_empty() {
                current.push_str(rest);
            }
            continue;
        }
        if !current.is_empty() && current.len() + 1 + word.len() > max_chunk_chars {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_paragraphs_up_to_the_target_size() {
        let text = "para one\n\npara two\n\npara three";
        let chunks = chunk_text(text, 100);
        assert_eq!(chunks, vec!["para one\n\npara two\n\npara three".to_string()]);

        let tight_chunks = chunk_text(text, 9);
        assert_eq!(tight_chunks, vec!["para one".to_string(), "para two".to_string(), "para".to_string(), "three".to_string()]);
        assert!(tight_chunks.iter().all(|c| c.len() <= 9), "{tight_chunks:?}");
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("   \n\n  ", 100).is_empty());
    }

    #[test]
    fn splits_a_paragraph_too_big_for_the_limit_on_its_own() {
        let paragraph = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let chunks = chunk_text(paragraph, 20);
        assert!(chunks.iter().all(|c| c.len() <= 20), "{chunks:?}");
        assert_eq!(chunks.join(" "), paragraph);
    }

    #[test]
    fn hard_splits_a_single_word_longer_than_the_limit_at_a_char_boundary() {
        // multi-byte UTF-8 throughout, so a naive byte-offset split would panic
        let word = "\u{00e9}".repeat(10); // 20 bytes, 10 chars
        let chunks = chunk_text(&word, 6);
        assert!(chunks.iter().all(|c| c.len() <= 6), "{chunks:?}");
        assert_eq!(chunks.concat(), word);
    }
}
