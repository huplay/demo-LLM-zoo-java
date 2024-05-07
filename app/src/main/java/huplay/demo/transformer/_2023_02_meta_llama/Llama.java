package huplay.demo.transformer._2023_02_meta_llama;

import huplay.demo.transformer.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.TransformerUtil.RMSLayerNorm;
import static huplay.demo.config.ParameterType.*;

/**
  Meta Llama transformer

  Differences to GPT-2:
    - Rotary Position Embedding (RoPE)
    - Optionally Grouped Query Attention (GQA) (Only at certain models)
    - Two separate MLP layers, results multiplied and processed by a third layer
    - SwiGLU activation function
    - RMS normalisation
    - No biases, only weights
    - Query, key and value matrices are stored separately

 * @author Hunor Szegi
 */
public class Llama extends BaseTransformer
{
    public Llama(Config config)
    {
        super(config, DecoderType.META_LLAMA);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "embed_tokens.weight", embeddingCount + 3, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "norm.weight", hiddenSize);
    }

    public float[] execute(int pos, int token, boolean isOutputProcessing)
    {
        // Find the embeddings of the token
        float[] hiddenState = matrix(TOKEN_EMBEDDINGS)[token];

        // Decoder stack
        for (BaseDecoder decoder : decoders)
        {
            hiddenState = decoder.execute(hiddenState, isOutputProcessing);
        }

        // Final normalization
        if (isOutputProcessing) // No need to execute for input tokens
        {
            hiddenState = RMSLayerNorm(hiddenState, vector(OUTPUT_NORM_WEIGHT), epsilon);
        }

        return hiddenState;
    }
}
