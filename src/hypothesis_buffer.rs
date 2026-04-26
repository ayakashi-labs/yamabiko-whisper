//! LocalAgreement-2 hypothesis buffer.
//!
//! Port of the `HypothesisBuffer` from `ufal/whisper_streaming`
//! (Macháček et al., "Turning Whisper into Real-Time Transcription System",
//! IJCNLP-AACL 2023 Demo). The buffer commits a word the moment two
//! consecutive hypotheses agree on its prefix position; this is the core
//! latency / quality trade-off of LocalAgreement-n with n = 2.

/// One recognised word with absolute (offset-applied) timestamps in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Rolling buffer that turns a sequence of overlapping Whisper hypotheses
/// into a monotonic stream of committed words.
#[derive(Default)]
pub struct HypothesisBuffer {
    /// All words committed so far that are still inside the current audio
    /// window (i.e. not yet `pop_committed`-ed away).
    pub committed_in_buffer: Vec<Word>,
    /// Tentative hypothesis from the previous `insert` call. Compared head-
    /// to-head against `new` in `flush` to find the agreed prefix.
    pub buffer: Vec<Word>,
    /// Hypothesis just staged by the latest `insert` call, after offset
    /// shifting and de-duplication.
    pub new: Vec<Word>,
    /// End time of the last committed word; new words at or before this time
    /// (with a 0.1 s tolerance) are dropped on insert.
    pub last_committed_time: f64,
    /// Text of the last committed word (informational, kept to match the
    /// reference implementation).
    pub last_committed_word: Option<String>,
}

impl HypothesisBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a new hypothesis. `offset` is added to every word's timestamps
    /// so that callers can pass per-iteration relative timings.
    ///
    /// Two filtering steps follow the offset shift:
    /// 1. Drop words that start at or before `last_committed_time - 0.1` —
    ///    they are already in the past and would re-emit committed audio.
    /// 2. If the first new word starts within ±1 s of `last_committed_time`,
    ///    look for the longest n-gram (n = 1..=5) match between the tail of
    ///    `committed_in_buffer` and the head of `new`, and drop the matching
    ///    head from `new`. This suppresses duplicates that appear when the
    ///    init prompt drags previously committed words back into the model
    ///    output.
    pub fn insert(&mut self, new: Vec<Word>, offset: f64) {
        let shifted: Vec<Word> = new
            .into_iter()
            .map(|w| Word {
                start: w.start + offset,
                end: w.end + offset,
                text: w.text,
            })
            .collect();

        let mut filtered: Vec<Word> = shifted
            .into_iter()
            .filter(|w| w.start > self.last_committed_time - 0.1)
            .collect();

        if let Some(first) = filtered.first()
            && (first.start - self.last_committed_time).abs() < 1.0
            && !self.committed_in_buffer.is_empty()
        {
            let cn = self.committed_in_buffer.len();
            let nn = filtered.len();
            let max_n = cn.min(nn).min(5);
            // Reference behaviour (whisper_streaming): try i = 1..=max_n
            // and break on the first match — i.e. drop the shortest
            // n-gram of duplicates, not the longest.
            for i in 1..=max_n {
                let tail = self.committed_in_buffer[cn - i..]
                    .iter()
                    .map(|w| w.text.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                let head = filtered[..i]
                    .iter()
                    .map(|w| w.text.clone())
                    .collect::<Vec<_>>()
                    .join(" ");
                if tail == head {
                    filtered.drain(0..i);
                    break;
                }
            }
        }

        self.new = filtered;
    }

    /// Commit the longest common prefix between `new` and the previous
    /// hypothesis kept in `buffer`. This is LocalAgreement-2: a word is
    /// emitted only after it has appeared in the same position in two
    /// consecutive hypotheses.
    pub fn flush(&mut self) -> Vec<Word> {
        let mut commit: Vec<Word> = Vec::new();
        while !self.new.is_empty() && !self.buffer.is_empty() {
            if self.new[0].text == self.buffer[0].text {
                let w = self.new.remove(0);
                self.buffer.remove(0);
                self.last_committed_time = w.end;
                self.last_committed_word = Some(w.text.clone());
                self.committed_in_buffer.push(w.clone());
                commit.push(w);
            } else {
                break;
            }
        }
        // Whatever survives in `new` becomes the next hypothesis to compare
        // against. `new` is then cleared so the next `insert` starts fresh.
        self.buffer = std::mem::take(&mut self.new);
        commit
    }

    /// Drop committed words that ended at or before `time`. Used after the
    /// audio buffer is trimmed, to keep `committed_in_buffer` aligned with
    /// the still-resident audio window.
    pub fn pop_committed(&mut self, time: f64) {
        while let Some(first) = self.committed_in_buffer.first() {
            if first.end <= time {
                self.committed_in_buffer.remove(0);
            } else {
                break;
            }
        }
    }

    /// Snapshot of the still-tentative buffer (the hypothesis kept around
    /// for the next agreement check). Useful for rendering an "in-flight"
    /// preview to the user.
    pub fn complete(&self) -> Vec<Word> {
        self.buffer.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start: f64, end: f64, text: &str) -> Word {
        Word {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn prefix_match_commits() {
        let mut hb = HypothesisBuffer::new();
        // Seed buffer (prior hypothesis) by feeding it through insert+flush.
        hb.insert(
            vec![w(0.0, 0.5, "a"), w(0.5, 1.0, "b"), w(1.0, 1.5, "c")],
            0.0,
        );
        let first = hb.flush();
        assert!(
            first.is_empty(),
            "first insert has no buffer to match against"
        );

        // Second insert agrees on a, b but diverges at c -> x.
        hb.insert(
            vec![w(0.0, 0.5, "a"), w(0.5, 1.0, "b"), w(1.0, 1.5, "x")],
            0.0,
        );
        let committed = hb.flush();
        assert_eq!(committed, vec![w(0.0, 0.5, "a"), w(0.5, 1.0, "b")]);
        assert_eq!(hb.buffer, vec![w(1.0, 1.5, "x")]);
        assert_eq!(hb.last_committed_time, 1.0);
        assert_eq!(hb.last_committed_word.as_deref(), Some("b"));
    }

    #[test]
    fn first_token_mismatch_commits_nothing() {
        let mut hb = HypothesisBuffer::new();
        hb.insert(vec![w(0.0, 0.5, "a"), w(0.5, 1.0, "b")], 0.0);
        hb.flush();
        hb.insert(vec![w(0.0, 0.5, "X"), w(0.5, 1.0, "Y")], 0.0);
        let committed = hb.flush();
        assert!(committed.is_empty());
        assert_eq!(hb.buffer, vec![w(0.0, 0.5, "X"), w(0.5, 1.0, "Y")]);
    }

    #[test]
    fn drops_words_before_last_committed_time() {
        let mut hb = HypothesisBuffer::new();
        hb.last_committed_time = 5.0;
        // 4.5 < 5.0 - 0.1 -> drop. 4.95 > 4.9 -> keep. 6.0 -> keep.
        hb.insert(
            vec![w(4.5, 4.7, "old"), w(4.95, 5.2, "edge"), w(6.0, 6.5, "new")],
            0.0,
        );
        assert_eq!(hb.new, vec![w(4.95, 5.2, "edge"), w(6.0, 6.5, "new")]);
    }

    #[test]
    fn dedupe_ngram_overlap() {
        let mut hb = HypothesisBuffer::new();
        hb.committed_in_buffer = vec![
            w(0.0, 0.4, "the"),
            w(0.4, 0.8, "quick"),
            w(0.8, 1.2, "brown"),
        ];
        hb.last_committed_time = 1.2;
        // First word starts within ±1 s of 1.2; head matches tail of length 2.
        hb.insert(
            vec![
                w(1.3, 1.6, "quick"),
                w(1.6, 2.0, "brown"),
                w(2.0, 2.4, "fox"),
            ],
            0.0,
        );
        assert_eq!(hb.new, vec![w(2.0, 2.4, "fox")]);
    }

    #[test]
    fn pop_committed_drops_old() {
        let mut hb = HypothesisBuffer::new();
        hb.committed_in_buffer = vec![w(0.0, 1.0, "a"), w(1.0, 2.0, "b"), w(2.0, 3.0, "c")];
        hb.pop_committed(2.0);
        assert_eq!(hb.committed_in_buffer, vec![w(2.0, 3.0, "c")]);
    }

    #[test]
    fn complete_returns_buffer_clone() {
        let mut hb = HypothesisBuffer::new();
        hb.buffer = vec![w(0.0, 0.5, "a")];
        let snapshot = hb.complete();
        hb.buffer.clear();
        assert_eq!(snapshot, vec![w(0.0, 0.5, "a")]);
    }
}
