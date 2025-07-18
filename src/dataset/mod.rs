use burn::data::dataset::Dataset;

#[derive(Debug, Clone)]
pub struct TokenPair {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
}

#[derive(Debug)]
pub struct TokenPairDataset {
    data: Vec<TokenPair>,
}

impl TokenPairDataset {
    pub fn from_tokens(tokens: Vec<i32>, block_size: usize) -> (Self, Self) {
        let mut train_data = tokens;

        let split_at = (train_data.len() as f64 * 0.9) as usize;
        let test_data = train_data.split_off(split_at);

        // [1,2,3,4,5,6] = [[1,2,3,4],[2,3,4,5],[3,4,5,6],...] = [[[1,2,3,4]=input,[2,3,4,5]=target],[[2,3,4,5],[3,4,5,6]]]
        // [1,2,3,4] = input
        // [2,3,4,5] = target
        let train_dataset = Self {
            data: train_data
                .windows(block_size)
                .collect::<Vec<_>>()
                .windows(2)
                .map(|v| TokenPair {
                    input: v[0].to_vec(),
                    target: v[1].to_vec(),
                })
                .collect(),
        };

        let test_dataset = Self {
            data: test_data
                .windows(block_size)
                .collect::<Vec<_>>()
                .windows(2)
                .map(|v| TokenPair {
                    input: v[0].to_vec(),
                    target: v[1].to_vec(),
                })
                .collect(),
        };

        (train_dataset, test_dataset)
    }
}

impl Dataset<TokenPair> for TokenPairDataset {
    fn get(&self, index: usize) -> Option<TokenPair> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::TokenPairDataset;

    #[test]
    pub fn test_ok() {
        let (train, test) = TokenPairDataset::from_tokens(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4,
                5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9,
            ], //
            4,
        );

        println!("train dataset: {:?}", train);
        println!("test dataset: {:?}", test);
    }
}
