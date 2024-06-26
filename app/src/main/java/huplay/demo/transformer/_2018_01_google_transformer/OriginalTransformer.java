package huplay.demo.transformer._2018_01_google_transformer;

import huplay.demo.config.Config;
import huplay.demo.transformer.DecoderType;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.util.Vector;

import static huplay.demo.config.ParameterType.TOKEN_EMBEDDINGS;

/**
  Google Brain, the original decoder-only Transformer

  The encoder-decoder architecture was described in the famous "Attention Is All You Need" paper:
  June 2017: https://arxiv.org/abs/1706.03762

  The decoder-only variant was described in "Generating Wikipedia by Summarizing Long Sequences"
  Jan 2018: https://arxiv.org/abs/1801.10198

  Features:
    - Sinusoid position embedding added to the input at the beginning
    - Normalization is used at the end of the attention and feed-forward blocks
    - Residual connections at the attention and feed-forward blocks
    - Multi-head attention
    - Score dividend in the attention, which is calculated as sqrt(headSize)
    - Single layer projection at the end of the attention blocks
    - Feed-forward block has two layers (layer1: 4 * hiddenSize neurons, layer2: hiddenSize neurons)
    - GELU activation function (used only at the first feed-forward layer)
    - 32 bit parameters
    - query/key/value matrices are stored in a single matrix

   The weights of the trained model wasn't published.
   I don't know about any model which uses exactly the same architecture, so

 * @author Hunor Szegi
 */
public class OriginalTransformer extends BaseTransformer
{
    private final float[][] positionMatrix;

    public OriginalTransformer(Config config)
    {
        super(config, DecoderType.ORIGINAL_DECODER);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "tokens_embed.weight", tokenCount, hiddenSize);

        // Calculates the sinusoidal transform matrix for the position embedding
        this.positionMatrix = calculatePositionMatrix();
    }

    public Vector execute(int pos, int token, boolean isOutputProcessing)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS)[token];

        // Position embedding
        for (int i = 0; i < hiddenState.size(); i++)
        {
            hiddenState.set(i, hiddenState.get(i) * positionMatrix[pos][i]);
        }

        // Decoder stack
        for (BaseDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        return hiddenState;
    }

    private float[][] calculatePositionMatrix()
    {
        float[][] positionMatrix = new float[contextSize][hiddenSize];

        float[] positions = new float[contextSize];
        for (int i = 0; i < contextSize; i++)
        {
            positions[i] = i;
        }

        float[] progression = new float[contextSize / 2];
        for (int i = 0; i < contextSize / 2; i++)
        {
            progression[i] = (float) Math.exp(-i * Math.log(10000) / contextSize);
        }

        for (int pos = 0; pos < contextSize; pos++)
        {
            for (int k = 0; k < hiddenSize / 2; k++)
            {
                int i = 2 * k;
                positionMatrix[pos][i] = (float) Math.sin(positions[i] * progression[k]);
                positionMatrix[pos][i + 1] = (float) Math.sin(positions[i + 1] * progression[k]);
            }
        }

        return positionMatrix;
    }
}
