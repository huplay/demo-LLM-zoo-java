package huplay.demo.transformer.meta.llama;

import huplay.demo.config.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.AbstractTransformer;
import huplay.demo.transformer.AbstractDecoder;

import static huplay.demo.TransformerUtil.layerNorm;
import static huplay.demo.config.ParameterType.*;

/**
 * Meta Llama transformer
 * -
 * Differences to GPT-2:
 * - Rotary Position Embedding (RoPE)
 * - Grouped Query Attention (GQA) (Only at certain models)
 * - Two separate MLP layers, results multiplied and processed by a third layer
 * - SwiGLU activation function
 * - RMS normalisation
 * - No biases, only weights
 * - Query, key and value matrices are stored separately
 */
public class Llama extends AbstractTransformer
{
    public Llama(Config config)
    {
        super(config, true ? DecoderType.META_LLAMA_MHA : DecoderType.META_LLAMA_GQA);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "embed_tokens.weight", tokenCount, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "norm.weight", hiddenSize);
    }

    public float[] execute(int pos, float[] hiddenState, boolean isOutputProcessing)
    {
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
