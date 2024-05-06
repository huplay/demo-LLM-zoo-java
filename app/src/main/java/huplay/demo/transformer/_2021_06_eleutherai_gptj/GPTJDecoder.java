package huplay.demo.transformer._2021_06_eleutherai_gptj;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;

/**
 * EleutherAI GPT-J decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTJDecoder extends BaseDecoder
{
    private final int maxAttentionSize;

    public GPTJDecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        // Load parameters
        // attn.bias BOOL: 1x1x2048x2048

        loadVector(ATT_NORM_WEIGHT, "transformer.h.{decoderId}.ln_1.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "transformer.h.{decoderId}.ln_1.bias", hiddenSize);
        loadMatrix(ATT_QUERY_WEIGHT, "transformer.h.{decoderId}.attn.q_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_KEY_WEIGHT, "transformer.h.{decoderId}.attn.k_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_VALUE_WEIGHT, "transformer.h.{decoderId}.attn.v_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_PROJ_WEIGHT, "transformer.h.{decoderId}.attn.out_proj.weight", hiddenSize, hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "transformer.ln_f.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "transformer.ln_f.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "transformer.h.{decoderId}.mlp.fc_in.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_1_BIAS, "transformer.h.{decoderId}.mlp.fc_in.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "transformer.h.{decoderId}.mlp.fc_out.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_2_BIAS, "transformer.h.{decoderId}.mlp.fc_out.bias", hiddenSize);

        maxAttentionSize = 256; // TODO: Move sparse attention to logic, not as config
    }

    public float[] execute(float[] hiddenState, boolean isOutputProcessing)
    {
        // Attention block
        hiddenState = attentionBlock(hiddenState);

        // Feed-forward block
        if (isOutputProcessing || ! lastDecoder) // No need to execute for input tokens at the last decoder
        {
            hiddenState = feedForwardBlock(hiddenState);
        }

        return hiddenState;
    }

    private float[] attentionBlock(float[] inputHiddenState)
    {
        // Normalisation
        float[] hiddenState = layerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private float[] feedForwardBlock(float[] inputHiddenState)
    {
        // Normalisation
        //float[] hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        float[] hiddenState = neuralLayers(inputHiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        float[] query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));
        float[] key = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        float[] value = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        float[][] queryByHead = UTIL.splitVector(query, headCount);
        float[][] keyByHead = UTIL.splitVector(key, headCount);
        float[][] valueByHead = UTIL.splitVector(value, headCount);

        // Position embedding (RoPE)
        applyPosition(query, key);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Used only at sparse attention:
        /*if (storedSize > maxAttentionSize)
        {
            // Topping the maximum attention size we can drop the oldest stored values
            storedKeys.remove(0);
            storedValues.remove(0);
        }*/

        // Declaration of the variable for collecting the attention results for all heads
        float[][] valueAggregate = new float[headCount][headSize];

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Calculate the scores
            float[] actualQuery = queryByHead[head];
            float[] scores = new float[storedSize];

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                float[] relatedKey = storedKeys.get(pos)[head];
                scores[pos] = UTIL.dotProduct(actualQuery, relatedKey);
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                float[] relatedValue = storedValues.get(pos)[head];
                float[] multipliedValue = UTIL.mulVectorByScalar(relatedValue, scores[pos]);
                valueAggregate[head] = UTIL.addVectors(valueAggregate[head], multipliedValue);
            }
        }

        // Concatenate the results for all heads
        hiddenState = UTIL.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        //hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }

    private float[] neuralLayers(float[] hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState[neuron] = TransformerUtil.gelu(hiddenState[neuron]);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }

    protected void applyPosition(float[] query, float[] key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query[i];
            query[i] = query0 * x - query[i + 1] * y;
            query[i + 1] = query0 * y - query[i + 1] * x;

            // Rotate key
            float key0 = key[i];
            key[i] = key0 * x - key[i + 1] * y;
            key[i + 1] = key0 * y - key[i + 1] * x;
        }
    }
}
