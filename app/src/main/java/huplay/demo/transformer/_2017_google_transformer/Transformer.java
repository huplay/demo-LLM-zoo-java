package huplay.demo.transformer._2017_google_transformer;

import huplay.demo.config.Config;
import huplay.demo.config.DecoderType;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.transformer.BaseTransformer;

import static huplay.demo.config.ParameterType.TOKEN_EMBEDDINGS;

/**
 * Google Brain, the original transformer - Described in the "Attention Is All You Need" paper
 * -
 * Features:
 * - Sinusoid position embedding added to the input at the beginning
 * - Normalization is used at the end of the attention and feed-forward blocks
 * - Residual connections at the attention and feed-forward blocks
 * - Multi-head attention
 * - Score dividend in the attention, which is calculated as sqrt(headSize)
 * - Single layer projection at the end of the attention blocks
 * - Feed-forward block has two layers (layer1: 4 * hiddenSize neurons, layer2: hiddenSize neurons)
 * - GELU activation function (used only at the first feed-forward layer)
 * - 32 bit parameters
 * - query/key/value matrices are stored in a single matrix
 */
public class Transformer extends BaseTransformer
{
    private final float[][] transformMatrix;

    public Transformer(Config config)
    {
        super(config, DecoderType.OPENAI_GPT_1);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "tokens_embed.weight", tokenCount, hiddenSize);

        // Calculates the sinusoidal transform matrix for the position embedding
        this.transformMatrix = calculateTransformMatrix(config.getMaxLength(), config.getHiddenSize());
    }

    public float[] execute(int pos, float[] hiddenState, boolean isOutputProcessing)
    {
        // Position embedding
        for (int i = 0; i < hiddenState.length; i++)
        {
            hiddenState[i] = hiddenState[i] * transformMatrix[pos][i];
        }

        // Decoder stack
        for (BaseDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        return hiddenState;
    }

    private float[][] calculateTransformMatrix(int maxLength, int hiddenSize)
    {
        float[][] transformMatrix = new float[maxLength][hiddenSize];

        float[] positions = new float[maxLength];
        for (int i = 0; i < maxLength; i++)
        {
            positions[i] = i;
        }

        float[] progression = new float[maxLength / 2];
        for (int i = 0; i < maxLength / 2; i++)
        {
            progression[i] = (float) Math.exp(-i * Math.log(10000) / maxLength);
        }

        for (int pos = 0; pos < maxLength; pos++)
        {
            for (int k = 0; k < hiddenSize / 2; k++)
            {
                int i = 2 * k;
                transformMatrix[pos][i] = (float) Math.sin(positions[i] * progression[k]);
                transformMatrix[pos][i + 1] = (float) Math.sin(positions[i + 1] * progression[k]);
            }
        }

        return transformMatrix;
    }
}
