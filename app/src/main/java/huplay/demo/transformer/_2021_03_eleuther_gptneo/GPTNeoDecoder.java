package huplay.demo.transformer._2021_03_eleuther_gptneo;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;
import static huplay.demo.util.Vector.newVectorArray;

/**
 * EleutherAI GPT-NEO decoder implementation
 *
 * @author Hunor Szegi
 */
public class GPTNeoDecoder extends BaseDecoder
{
    private final int maxAttentionSize;

    public GPTNeoDecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        // Load parameters
        loadVector(ATT_NORM_WEIGHT, "ln_1.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "ln_1.bias", hiddenSize);
        loadMatrix(ATT_QUERY_WEIGHT, "attn.attention.q_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_KEY_WEIGHT, "attn.attention.k_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_VALUE_WEIGHT, "attn.attention.v_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_PROJ_WEIGHT, "attn.attention.out_proj.weight", hiddenSize, hiddenSize);
        loadVector(ATT_PROJ_BIAS, "attn.attention.out_proj.bias", hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "ln_2.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "ln_2.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.c_fc.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_1_BIAS, "mlp.c_fc.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.c_proj.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_2_BIAS, "mlp.c_proj.bias", hiddenSize);

        maxAttentionSize = 256; // TODO: Move sparse attention to logic, not as config
    }

    public Vector execute(Vector hiddenState, boolean isOutputProcessing)
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

    private Vector attentionBlock(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = layerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector feedForwardBlock(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        hiddenState = neuralLayers(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));
        Vector key = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        Vector value = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        Vector[] queryByHead = UTIL.splitVector(query, headCount);
        Vector[] keyByHead = UTIL.splitVector(key, headCount);
        Vector[] valueByHead = UTIL.splitVector(value, headCount);

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
        Vector[] valueAggregate = newVectorArray(hiddenState.getFloatType(), headCount, headSize);

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            // Calculate the scores
            Vector actualQuery = queryByHead[head];
            Vector scores = new Vector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos)[head];
                scores.set(pos, UTIL.dotProduct(actualQuery, relatedKey));
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos)[head];
                Vector multipliedValue = UTIL.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate[head] = UTIL.addVectors(valueAggregate[head], multipliedValue);
            }
        }

        // Concatenate the results for all heads
        hiddenState = UTIL.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }

    private Vector neuralLayers(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, TransformerUtil.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
