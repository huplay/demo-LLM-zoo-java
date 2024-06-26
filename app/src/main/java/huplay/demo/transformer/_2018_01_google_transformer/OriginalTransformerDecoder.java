package huplay.demo.transformer._2018_01_google_transformer;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;

/**
 * Decoder implementation of the original decoder-only Transformer architecture created by Google Brain
 *
 * @author Hunor Szegi
 */
public class OriginalTransformerDecoder extends BaseDecoder
{
    public OriginalTransformerDecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        // Load parameters
        loadVector(ATT_NORM_WEIGHT, "ln_1.weight", hiddenSize);
        loadVector(ATT_NORM_BIAS, "ln_1.bias", hiddenSize);
        loadMatrix(ATT_QUERY_KEY_VALUE_WEIGHT, "attn.c_attn.weight", hiddenSize, hiddenSize * 3);
        loadVector(ATT_QUERY_KEY_VALUE_BIAS, "attn.c_attn.bias", hiddenSize * 3);
        loadMatrix(ATT_PROJ_WEIGHT, "attn.c_proj.weight", hiddenSize, hiddenSize);
        loadVector(ATT_PROJ_BIAS, "attn.c_proj.bias", hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "ln_2.weight", hiddenSize);
        loadVector(MLP_NORM_BIAS, "ln_2.bias", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.c_fc.weight", hiddenSize, feedForwardSize);
        loadVector(MLP_1_BIAS, "mlp.c_fc.bias", feedForwardSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.c_proj.weight", feedForwardSize, hiddenSize);
        loadVector(MLP_2_BIAS, "mlp.c_proj.bias", hiddenSize);

        // Calculate the attention dividend
        attentionDividend = sqrt(headSize);
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
        // Attention
        Vector hiddenState = attention(inputHiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        // Normalisation
        hiddenState = layerNorm(hiddenState, vector(ATT_NORM_WEIGHT), vector(ATT_NORM_BIAS), epsilon);

        return hiddenState;
    }

    private Vector feedForwardBlock(Vector inputHiddenState)
    {
        // Neural layers
        Vector hiddenState = neuralLayers(inputHiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        //  Normalisation
        hiddenState = layerNorm(hiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        return hiddenState;
    }

    private Vector attention(Vector hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        Vector queryKeyValue = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = UTIL.addVectors(queryKeyValue, vector(ATT_QUERY_KEY_VALUE_BIAS));

        // Split the query/key/value
        Vector[] split = UTIL.splitVector(queryKeyValue, 3);
        Vector query = split[0];
        Vector key = split[1];
        Vector value = split[2];

        // Split the query, key and value vectors into pieces for all heads
        Vector[] queryByHead = UTIL.splitVector(query, headCount);
        Vector[] keyByHead = UTIL.splitVector(key, headCount);
        Vector[] valueByHead = UTIL.splitVector(value, headCount);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Declaration of the variable for collecting the attention results for all heads
        Vector[] valueAggregate = new Vector[headCount];

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
                float score = UTIL.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
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
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }

    private Vector neuralLayers(Vector hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState.set(neuron, TransformerUtil.gelu(hiddenState.get(neuron)));
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
