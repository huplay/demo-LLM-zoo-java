package huplay.demo.transformer._2021_06_eleuther_gptj;

import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.DecoderType;
import huplay.demo.util.Vector;

import static huplay.demo.TransformerUtil.layerNorm;
import static huplay.demo.config.ParameterType.*;

/**
 * TODO:
 * https://github.com/jzhang38/TinyLlama/issues/24
 *
 * GPTJ style: Original Llama, llama.cpp
 * rotates pairs of even and odd dimensions
 *
 * NEOX style: OpenLlama (all HF Llama)
 * rotates the 1st and 2nd half
 *
 * HF permutes the weight

  EleutherAI GPT-J transformer

  Differences to GPT-NEO:
    - Rotary Position Embedding (RoPE)
    - Uses bias at token embeddings
    - No bias at attention query/key/value matrices and projection (but has bias at the mlp component)
    - Feed-forward normalization parameters are common in all decoders, and the same used at final normalization
 * @author Hunor Szegi
 */
public class GPTJ extends BaseTransformer
{
    public GPTJ(Config config)
    {
        super(config, DecoderType.ELEUTHERAI_GPT_J);

        // Load parameters
        loadMatrix(TOKEN_EMBEDDINGS, "lm_head.weight", tokenCount, hiddenSize);
        loadVectorOptional(TOKEN_EMBEDDING_BIAS, "lm_head.bias", tokenCount); // TODO: This is new

        loadVector(OUTPUT_NORM_WEIGHT, "transformer.ln_f.weight", hiddenSize);
        loadVector(OUTPUT_NORM_BIAS, "transformer.ln_f.bias", hiddenSize);
    }

    public Vector execute(int pos, int token, boolean isOutputProcessing)
    {
        // Find the embeddings of the token
        Vector hiddenState = matrix(TOKEN_EMBEDDINGS)[token];
        //hiddenState = UTIL.addVectors(hiddenState, vector(TOKEN_EMBEDDING_BIAS));

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
