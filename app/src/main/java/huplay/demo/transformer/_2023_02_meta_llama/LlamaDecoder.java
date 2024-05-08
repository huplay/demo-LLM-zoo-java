package huplay.demo.transformer._2023_02_meta_llama;

import huplay.demo.TransformerUtil;
import huplay.demo.config.Config;
import huplay.demo.transformer.BaseDecoder;
import huplay.demo.util.Vector;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.TransformerUtil.*;
import static huplay.demo.config.ParameterType.*;
import static huplay.demo.util.Vector.newVectorArray;

/**
 * Meta Llama decoder implementation
 *
 * @author Hunor Szegi
 */
public class LlamaDecoder extends BaseDecoder
{
    private final int kvHeadCount;
    private final int kvHeadSize;

    public LlamaDecoder(Config config, int decoderId)
    {
        super(config, decoderId);

        // Load parameters
        loadMatrix(ATT_QUERY_WEIGHT, "self_attn.q_proj.weight", hiddenSize, hiddenSize);

        // Read the optional config for the "key/value head" count
        // (At the original attention (MHA) there was only a single kind of head. Call it "query head" from now on.)
        // If the "query head" is different to the "key/value head" count, we are using Grouped Query Attention (GQA)
        kvHeadCount = config.getIntOptional("num_key_value_heads", headCount);
        kvHeadSize = headCount / kvHeadCount;
        loadMatrix(ATT_KEY_WEIGHT, "self_attn.k_proj.weight", hiddenSize, hiddenSize / kvHeadSize);
        loadMatrix(ATT_VALUE_WEIGHT, "self_attn.v_proj.weight", hiddenSize, hiddenSize / kvHeadSize);

        loadVector(ATT_NORM_WEIGHT, "input_layernorm.weight", hiddenSize);
        loadMatrix(ATT_PROJ_WEIGHT, "self_attn.o_proj.weight", hiddenSize, hiddenSize);
        loadVector(MLP_NORM_WEIGHT, "post_attention_layernorm.weight", hiddenSize);
        loadMatrix(MLP_1_WEIGHT, "mlp.gate_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_2_WEIGHT, "mlp.up_proj.weight", feedForwardSize, hiddenSize);
        loadMatrix(MLP_3_WEIGHT, "mlp.down_proj.weight", hiddenSize, feedForwardSize);

        // Calculate the attention dividend
        this.attentionDividend = sqrt(headSize);
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
        Vector hiddenState = RMSLayerNorm(inputHiddenState, vector(ATT_NORM_WEIGHT), epsilon);

        if (kvHeadSize == 1)
        {
            // Multi Head Attention (MHA)
            hiddenState = attention(hiddenState);
        }
        else
        {
            // Grouped Query Attention (GQA)
            hiddenState = groupedQueryAttention(hiddenState);
        }

        // Residual connection
        hiddenState =  UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    private Vector feedForwardBlock(Vector inputHiddenState)
    {
        // Normalisation
        Vector hiddenState = RMSLayerNorm(inputHiddenState, vector(MLP_NORM_WEIGHT), epsilon);

        // Neural layers
        hiddenState = neuralLayers(hiddenState);

        // Residual connection
        hiddenState = UTIL.addVectors(inputHiddenState, hiddenState);

        return hiddenState;
    }

    protected Vector attention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));
        Vector key = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        Vector value = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        Vector[] queryByHead = UTIL.splitVector(query, headCount);
        Vector[] keyByHead = UTIL.splitVector(key, headCount);
        Vector[] valueByHead = UTIL.splitVector(value, headCount);

        // Position embedding (RoPE)
        applyPosition(query, key);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByHead);
        storedValues.add(valueByHead);
        int storedSize = storedKeys.size();

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
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));

        return hiddenState;
    }

    protected void applyPosition(Vector query, Vector key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query.get(i);
            query.set(i, query0 * x - query.get(i + 1) * y);
            query.set(i + 1, query0 * y - query.get(i + 1) * x);

            // Rotate key
            float key0 = key.get(i);
            key.set(i, key0 * x - key.get(i + 1) * y);
            key.set(i + 1, key0 * y - key.get(i + 1) * x);
        }
    }

    protected Vector groupedQueryAttention(Vector hiddenState)
    {
        // Calculate the query, key and value vectors for the actual token
        Vector query = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_QUERY_WEIGHT));

        // The key and value matrices are smaller (less head count) than the query matrix
        Vector key = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_KEY_WEIGHT));
        Vector value = UTIL.mulVectorByMatrix(hiddenState, matrix(ATT_VALUE_WEIGHT));

        // Split the query, key and value vectors into pieces for all heads
        Vector[] queryByHead = UTIL.splitVector(query, headCount);
        Vector[] keyByGroup = UTIL.splitVector(key, headCount / kvHeadSize);
        Vector[] valueByGroup = UTIL.splitVector(value, headCount / kvHeadSize);

        // Position embedding (RoPE)
        applyGroupedPosition(query, key);

        // Store the keys and values (these will be available while the following tokens will be processed)
        storedKeys.add(keyByGroup);
        storedValues.add(valueByGroup);
        int storedSize = storedKeys.size();

        // Declaration of the variable for collecting the attention results for all heads
        Vector[] valueAggregate = new Vector[headCount];

        // Scoring the previous tokens (including the actual), separately for all heads
        for (int head = 0; head < headCount; head++)
        {
            int group = head % kvHeadCount;

            // Calculate the scores
            Vector actualQuery = queryByHead[head];
            Vector scores = new Vector(actualQuery.getFloatType(), storedSize);

            for (int pos = 0; pos < storedSize; pos++)
            {
                // The score is calculated multiplying the "actual" query vector and the "related" key vector
                Vector relatedKey = storedKeys.get(pos)[group];
                float score = UTIL.dotProduct(actualQuery, relatedKey);

                // Divide the score by the attention dividend
                scores.set(pos, score / attentionDividend);
            }

            // Rescaling the scores to values between 0 and 1
            scores = softmax(scores);

            // Multiply the value matrices with the scores, and sum up
            for (int pos = 0; pos < storedSize; pos++)
            {
                Vector relatedValue = storedValues.get(pos)[group];
                Vector multipliedValue = UTIL.mulVectorByScalar(relatedValue, scores.get(pos));
                valueAggregate[head] = UTIL.addVectors(valueAggregate[head], multipliedValue);
            }
        }

        // Concatenate the results for all heads
        hiddenState = UTIL.flattenMatrix(valueAggregate);

        // Projection neural layer
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(ATT_PROJ_WEIGHT));

        return hiddenState;
    }

    protected void applyGroupedPosition(Vector query, Vector key)
    {
        for (int i = 0; i < hiddenSize; i += 2)
        {
            int modulus = i % headSize;

            double frequency = 1.0 / pow(10000.0f, (float) modulus / headSize);
            double degree = frequency * storedKeys.size();
            float x = cos(degree);
            float y = sin(degree);

            // Rotate query
            float query0 = query.get(i);
            query.set(i, query0 * x - query.get(i + 1) * y);
            query.set(i + 1, query0 * y - query.get(i + 1) * x);

            if (i < hiddenSize / kvHeadSize)
            {
                // Rotate key
                float key0 = key.get(i);
                key.set(i, key0 * x - key.get(i + 1) * y);
                key.set(i + 1, key0 * y - key.get(i + 1) * x);
            }
        }
    }

    private Vector neuralLayers(Vector hiddenState)
    {
        // Feed parallel two layers with the same input
        Vector hiddenState1 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_1_WEIGHT));
        Vector hiddenState2 = UTIL.mulVectorByTransposedMatrix(hiddenState, matrix(MLP_2_WEIGHT));

        // Use SwiGLU activation function on the gate layer (no activation function on the other)
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, TransformerUtil.swiglu(hiddenState1.get(neuron)));
        }

        // Multiply the two outputs
        for (int neuron = 0; neuron < feedForwardSize; neuron++)
        {
            hiddenState1.set(neuron, hiddenState1.get(neuron) * hiddenState2.get(neuron));
        }

        // Use the third layer (no activation function)
        hiddenState = UTIL.mulVectorByTransposedMatrix(hiddenState1, matrix(MLP_3_WEIGHT));

        return hiddenState;
    }
}
