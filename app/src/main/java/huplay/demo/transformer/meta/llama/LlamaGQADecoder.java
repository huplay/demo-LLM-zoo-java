package huplay.demo.transformer.meta.llama;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.AbstractDecoder;

import static huplay.demo.App.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;

public class LlamaGQADecoder extends AbstractDecoder
{
    public LlamaGQADecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        attentionDividend = sqrt(headSize);
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
        return UTIL.addVectors(inputHiddenState, hiddenState);
    }

    private float[] feedForwardBlock(float[] inputHiddenState)
    {
        // Normalisation
        float[] hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        hiddenState = neuralLayers(hiddenState);

        // Residual connection
        return UTIL.addVectors(inputHiddenState, hiddenState);
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        float[] query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));
        query = UTIL.addVectors(query, vector(ATT_QUERY_BIAS));

        float[] key = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        key = UTIL.addVectors(key, vector(ATT_KEY_BIAS));

        float[] value = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));
        value = UTIL.addVectors(value, vector(ATT_VALUE_BIAS));

        // Split the query, key and value vectors into pieces for all heads
        float[][] queryByHead = UTIL.splitVector(query, headCount);
        float[][] keyByHead = UTIL.splitVector(key, headCount);
        float[][] valueByHead = UTIL.splitVector(value, headCount);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Declaration of the variable for collecting the attention results for all heads
        float[][] valueAggregate = new float[headCount][headSize];

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            float[] actualQuery = queryByHead[head];

            // Calculate the scores
            float[] scores = new float[storedSize];

            for (int pos = 0; pos < storedSize; pos++)
            {
                float[] relatedKey = storedKeys.get(pos)[head];

                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                float score = UTIL.dotProduct(actualQuery, relatedKey);

                scores[pos] = score / attentionDividend;
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
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

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
}
