package huplay.demo.transformer._2021_03_eleuther_gptneo;

import huplay.demo.transformer.DecoderType;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.layerNorm;
import static huplay.demo.config.ParameterType.*;

/**
  EleutherAI GPT-NEO transformer

  Differences to GPT-2:
    - Sparse decoders: Every second decoder uses local attention, using only the previous 256 tokens
    - No biases for the attention query/key/value matrices
    - query/key/value matrices are stored separately
    - No attention dividend, so the score isn't divided by a fixed value
    - The weights are stored in transposed matrices (easier to execute the vector-matrix multiplication)

 * @author Hunor Szegi
 */
public class GPTNeo extends BaseTransformer
{
    public GPTNeo(Config config)
    {
        super(config, DecoderType.ELEUTHERAI_GPT_NEO);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "wte.weight", tokenCount, hiddenSize);
        loadMatrix(POSITION_EMBEDDINGS, "wpe.weight", contextSize, hiddenSize);
        loadVector(OUTPUT_NORM_WEIGHT, "ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "ln_f.bias", hiddenSize);
    }

    public Vector execute(int pos, int token, boolean isOutputProcessing)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS)[token];

        // Position embedding
        hiddenState = UTIL.addVectors(hiddenState, matrix(POSITION_EMBEDDINGS)[pos]);

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
