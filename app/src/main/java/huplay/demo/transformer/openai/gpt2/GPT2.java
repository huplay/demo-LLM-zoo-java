package huplay.demo.transformer.openai.gpt2;

import huplay.demo.config.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.AbstractTransformer;
import huplay.demo.transformer.AbstractDecoder;

import static huplay.demo.App.UTIL;
import static huplay.demo.TransformerUtil.layerNorm;
import static huplay.demo.config.ParameterType.*;

/**
 * OpenAI GPT-2 transformer
 * -
 * Differences to GPT-1:
 * - The normalization is used at the beginning of the attention and mlp blocks
 * - Final normalization is added after the last decoder
 * (The normalization before the first decoder's attention block gives more numerical stability at larger models.)
 */
public class GPT2 extends AbstractTransformer
{
    public GPT2(Config config)
    {
        super(config, DecoderType.OPENAI_GPT_2);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "wte.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "wpe.weight", maxLength, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public float[] execute(int pos, float[] embedding, boolean isOutputProcessing)
    {
        // Position embedding
        float[] hiddenState = UTIL.addVectors(embedding, matrix(POSITION_EMBEDDINGS)[pos]);

        // Decoder stack
        for (AbstractDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        // Final normalization
        if (isOutputProcessing) // No need to execute for input tokens
        {
            hiddenState = layerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), vector(OUTPUT_NORM_BIAS), epsilon);
        }

        return hiddenState;
    }
}
