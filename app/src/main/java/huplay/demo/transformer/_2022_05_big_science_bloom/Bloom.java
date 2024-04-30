package huplay.demo.transformer._2022_05_big_science_bloom;

import huplay.demo.config.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.TransformerUtil.layerNorm;
import static huplay.demo.config.ParameterType.*;

/**
  BLOOM transformer

  Differences to GPT-2:
    - ALiBi position embedding, applied in the attention block to the score
    - 16 bit parameters (BFLOAT16 for the 176B model, FLOAT16 for the others)
    - Additional normalization at input (it was necessary because of the FLOAT16 data type, but used at all models)
    - The weights are stored in transposed matrices (easier to execute the vector-matrix multiplication)
    - The values in the query/key/value matrices are ordered first by head, and second by type
    - The key and value vectors are stored separately for the heads

 * @author Hunor Szegi
 */
public class Bloom extends BaseTransformer
{
    // TODO: Maybe something isn't perfect here, the output looks good, but very ofter repeats itself.
    public Bloom(Config config)
    {
        super(config, DecoderType.BIG_SCIENCE_BLOOM);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "word_embeddings.weight", tokenCount, hiddenSize);
        loadVector(INPUT_NORM_WEIGHT, "word_embeddings_layernorm.weight", hiddenSize);
        loadVector(INPUT_NORM_BIAS, "word_embeddings_layernorm.bias", hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public float[] execute(int pos, float[] embedding, boolean isOutputProcessing)
    {
        // Input normalization
        float[] hiddenState = layerNorm(embedding,  vector(INPUT_NORM_WEIGHT), vector(INPUT_NORM_BIAS), epsilon);

        // Decoder stack
        for (BaseDecoder decoder : decoders)
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
