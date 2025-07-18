use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, activation::softmax},
};

use crate::modeling::BigramModel;

pub fn generate<B: Backend>(
    model: &BigramModel<B>,
    prompt: Vec<i32>,
    max_new_tokens: usize,
    device: &B::Device,
) -> Vec<i32> {
    use rand::distr::Distribution;
    // [0] ->[0,  1]   ->[0,  1,  2]
    // [0']->[0', 1']->[0', 1' ,2']

    let mut tokens = prompt;
    let mut rng = rand::rng();

    for i in tokens.len()..tokens.len() + max_new_tokens {
        let x = TensorData::new(tokens.clone(), [1, i]);
        let x = Tensor::<B, 2, Int>::from_data(x, device);

        let logits = model.forward(x);
        let logits = logits.slice([0..1, i - 1..i]);

        let props = softmax(logits, 2);
        let props = props.to_data().to_vec::<f32>().unwrap();

        let distr = rand::distr::weighted::WeightedIndex::new(&props).unwrap();
        let next_token = distr.sample(&mut rng) as i32;

        tokens.push(next_token);
    }

    tokens
}
