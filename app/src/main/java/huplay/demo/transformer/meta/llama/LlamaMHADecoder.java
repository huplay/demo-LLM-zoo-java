package huplay.demo.transformer.meta.llama;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;

public class LlamaMHADecoder extends BaseDecoder
{
    public LlamaMHADecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        // Load parameters
        loadVector(ATT_NORM_WEIGHT, "input_layernorm.weight", hiddenSize);
        loadMatrix(ATT_QUERY_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_KEY_WEIGHT, "self_attn.k_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_VALUE_WEIGHT, "self_attn.v_proj.weight", hiddenSize, hiddenSize);
        loadMatrix(ATT_PROJ_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "post_attention_layernorm.weight", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.gate_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.up_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_3_WEIGHT, "mlp.down_proj.weight", hiddenSize, feedForwardSize);

        loadVectorOptional(ROTARY_EMBEDDING, "self_attn.rotary_emb.inv_freq", headSize / 2);

        // Calculate the attention dividend
        this.attentionDividend = sqrt(headSize);
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
        float[] hiddenState = RMSLayerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), epsilon);

        // Attention
        hiddenState = attention(hiddenState);

        // Residual connection
        hiddenState =  UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private float[] feedForwardBlock(float[] inputHiddenState)
    {
        // Normalisation
        float[] hiddenState = RMSLayerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), epsilon);

        // Neural layers
        hiddenState = neuralLayers(hiddenState);

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

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

        // Position embedding
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency;
            if (vector(ROTARY_EMBEDDING) == null)
            {
                // No rotary embedding parameters, do the standard calculation
                frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            }
            else
            {
                // Use the rotary embedding parameters. TODO: Fix it
                modulus = i % (headSize/2);
                frequency = vector(ROTARY_EMBEDDING)[modulus];
            }

            double degree = frequency * storedSize;
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
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));

        return hiddenState;
    }

    private float[] neuralLayers(float[] hiddenState)
    {
        // Feed parallel two layers with the same input
        float[] hiddenState1 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        float[] hiddenState2 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1[neuron] = TransformerUtil.swiglu(hiddenState1[neuron]);
        }

        // Multiply the two outputs
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1[neuron] = hiddenState1[neuron] * hiddenState2[neuron];
        }

        // Use the third layer (no activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState1, matrix(MLP_3_WEIGHT));

        return hiddenState;
    }
}
