package huplay.demo.transformer._2019_openai_gpt2;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;

public class GPT2Decoder extends BaseDecoder
{
    public GPT2Decoder(Config config, int decoderId)
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
        float[] hiddenState = layerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), vector(MLP_NORM_BIAS), epsilon);

        // Neural layers
        hiddenState = neuralLayers(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private float[] attention(float[] hiddenState)
    {
        // Calculate the query-key-value vectors for the actual token
        float[] queryKeyValue = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_QUERY_KEY_VALUE_WEIGHT));
        queryKeyValue = UTIL.addVectors(queryKeyValue, vector(ATT_QUERY_KEY_VALUE_BIAS));

        // Split the query/key/value
        float[][] split = UTIL.splitVector(queryKeyValue, 3);
        float[] query = split[0];
        float[] key = split[1];
        float[] value = split[2];

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
            // Calculate the scores
            float[] actualQuery = queryByHead[head];
            float[] scores = new float[storedSize];

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                float[] relatedKey = storedKeys.get(pos)[head];
                float score = UTIL.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
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
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(ATT_PROJ_BIAS));

        return hiddenState;
    }

    private float[] neuralLayers(float[] hiddenState)
    {
        // Layer 1: <mlpSize> neurons (usually 4 * <hiddenSize>) (using a gelu activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_1_BIAS));

        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState[neuron] = TransformerUtil.gelu(hiddenState[neuron]);
        }

        // Layer 2: <hiddenSize> neurons (without activation function)
        hiddenState = UTIL.mulVectorByMatrix(hiddenState, matrix(MLP_2_WEIGHT));
        hiddenState = UTIL.addVectors(hiddenState, vector(MLP_2_BIAS));

        return hiddenState;
    }
}
