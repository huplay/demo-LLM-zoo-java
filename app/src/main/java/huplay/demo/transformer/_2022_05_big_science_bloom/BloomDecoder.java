package huplay.demo.transformer._2022_05_big_science_bloom;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import java.util.ArrayList;
import java.util.List;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;
import static huplay.demo.util.Vector.newVectorArray;

/**
 * BLOOM decoder implementation
 *
 * @author Hunor Szegi
 */
public class BloomDecoder extends BaseDecoder
{
    private final float[] positionSlope;

    protected final List<List<Vector>> storedKeys = new ArrayList<>(headCount);
    protected final List<List<Vector>> storedValues = new ArrayList<>(headCount);

    public BloomDecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        for (int i = 0; i < headCount; i++)
        {
            storedKeys.add(new ArrayList<>());
            storedValues.add(new ArrayList<>());
        }

        // Load parameters
        loadVector(ATT_NORM_WEIGHT, "input_layernorm.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "input_layernorm.bias", hiddenSize);
        loadMatrix(ATT_QUERY_KEY_VALUE_WEIGHT, "self_attention.query_key_value.weight", hiddenSize * 3, hiddenSize);
        loadVector(ATT_QUERY_KEY_VALUE_BIAS, "self_attention.query_key_value.bias", hiddenSize * 3);
        loadMatrix(ATT_PROJ_WEIGHT, "self_attention.dense.weight", hiddenSize, hiddenSize);
        loadVector(ATT_PROJ_BIAS, "self_attention.dense.bias", hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "post_attention_layernorm.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "post_attention_layernorm.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.dense_h_to_4h.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_1_BIAS, "mlp.dense_h_to_4h.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.dense_4h_to_h.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_2_BIAS, "mlp.dense_4h_to_h.bias", hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);

        // Calculate the slope for the position embedding
        positionSlope = new float[headCount];
        float step = 1f / headCount;
        for (int i = 0; i < headCount; i++)
        {
            positionSlope[i] = step * (i + 1);
        }
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
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = UTIL.addVectors(queryKeyValue, vector(ATT_QUERY_KEY_VALUE_BIAS));

        // Split the query, key and value vectors into pieces for all heads
        Vector[] queryKeyValuesByHead = UTIL.splitVector(queryKeyValue, headCount);

        // Declaration of the variable for collecting the attention results for all heads
        Vector[] valueAggregate = newVectorArray(hiddenState.getFloatType(), headCount, headSize);

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            Vector queryKeyValueByHead = queryKeyValuesByHead[head];

            // Split the query/key/value
            Vector[] split = UTIL.splitVector(queryKeyValueByHead, 3);
            Vector queryByHead = split[0];
            Vector keyByHead = split[1];
            Vector valueByHead = split[2];

            storedKeys.get(head).add(keyByHead);
            storedValues.get(head).add(valueByHead);

            // Store the keys and values (these will be available while the following tokens will be processed)
            int storedSize = storedKeys.get(head).size();

            // Calculate the scores
            Vector scores = new Vector(hiddenState.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(head).get(pos);
                float score = UTIL.dotProduct(queryByHead, relatedKey);

                // Position embedding at score
                score = score - positionSlope[head] * (storedSize - pos - 1);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(head).get(pos);
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
