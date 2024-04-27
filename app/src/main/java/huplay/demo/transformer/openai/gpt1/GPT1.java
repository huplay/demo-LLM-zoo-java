package huplay.demo.transformer.openai.gpt1;

import huplay.demo.config.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.config.ParameterType.*;

/**
 * OpenAI GPT-1 transformer
 * -
 * Features:
 * - Learned position embedding added to the input at the beginning
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
public class GPT1 extends BaseTransformer
{
    public GPT1(Config config)
    {
        super(config, DecoderType.OPENAI_GPT_1);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "tokens_embed.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "positions_embed.weight", maxLength, hiddenSize);
    }

    public float[] execute(int pos, float[] embedding, boolean isOutputProcessing)
    {
        // Position embedding
        float[] hiddenState = UTIL.addVectors(embedding, matrix(POSITION_EMBEDDINGS)[pos]);

        // Decoder stack
        for (BaseDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        return hiddenState;
    }
}
