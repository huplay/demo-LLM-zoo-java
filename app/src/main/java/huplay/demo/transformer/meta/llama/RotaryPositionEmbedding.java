package huplay.demo.transformer.meta.llama;

import huplay.demo.config.Config;

public class RotaryPositionEmbedding extends SinusoidalPositionEmbedding
{
    private final int maxLength;

    public RotaryPositionEmbedding(Config config)
    {
        super(config);
        this.maxLength = config.getMaxLength();
    }

    @Override
    public float[] toInput(float[] input, int pos)
    {
        return input;
    }

    public float[] toQuery(float[] input, int length, int pos, int head)
    {
        // TODO: Implement the RoPE


    /* https://github.com/ZhuiyiTechnology/roformer:
    sinusoidal_pos.shape = [1, seq_len, hidden_size] # Sinusoidal position embeddings
    qw.shape = [batch_size, seq_len, num_heads, hidden_size]  # query hiddens
    kw.shape = [batch_size, seq_len, num_heads, hidden_size]  # key hiddens

    cos_pos = repeat_elements(sinusoidal_pos[..., None, 1::2], rep=2, axis=-1)
    sin_pos = repeat_elements(sinusoidal_pos[..., None, ::2], rep=2, axis=-1)

    qw2 = stack([-qw[..., 1::2], qw[..., ::2]], 4)
    qw2 = reshape(qw2, shape(qw))
    qw = qw * cos_pos + qw2 * sin_pos

    kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
    kw2 = K.reshape(kw2, K.shape(kw))
    kw = kw * cos_pos + kw2 * sin_pos

    # Attention
    a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
    */





    /*
    pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
    qw, kw = apply_rotary_position_embeddings(pos, qw, kw)

    import keras.backend as K


    def apply_rotary_position_embeddings(sinusoidal, *tensors):
        ndim = K.ndim(tensors[0])
        sinusoidal = align(sinusoidal, [0, 1, -1], ndim)
        cos_pos = K.repeat_elements(sinusoidal[..., 1::2], 2, -1)
        sin_pos = K.repeat_elements(sinusoidal[..., ::2], 2, -1)
        outputs = []
        for tensor in tensors:
            tensor2 = K.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
            tensor2 = K.reshape(tensor2, K.shape(tensor))
            outputs.append(tensor * cos_pos + tensor2 * sin_pos)
        return outputs[0] if len(outputs) == 1 else outputs

    def align(tensor, axes, ndim=None):
        """重新对齐tensor（批量版expand_dims）
        axes：原来的第i维对齐新tensor的第axes[i]维；
        ndim：新tensor的维度。
        """
        assert len(axes) == K.ndim(tensor)
        assert ndim or min(axes) >= 0
        ndim = ndim or max(axes) + 1
        indices = [None] * ndim
        for i in axes:
            indices[i] = slice(None)
        return tensor[indices]


     */


        // https://arxiv.org/abs/2104.09864
        // https://huggingface.co/docs/transformers/model_doc/roformer
        // https://github.com/ZhuiyiTechnology/roformer/commits/main
        // https://github.com/lucidrains/rotary-embedding-torch/commits/main
        // https://nn.labml.ai/transformers/rope/index.html

        return input;
    }
}
