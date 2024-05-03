package huplay.demo;

import huplay.demo.util.IndexedValue;
import huplay.demo.config.Config;
import huplay.demo.config.ParameterType;
import huplay.demo.tokenizer.Tokenizer;
import huplay.demo.transformer.BaseTransformer;

import java.util.*;
import static huplay.demo.AppMain.OUT;
import static huplay.demo.AppLoader.UTIL;

/**
 * Decoder-only Transformer implementation
 */
public class Generate
{
    private final Config config;
    private final Tokenizer tokenizer;
    private final BaseTransformer transformer;
    private final float[][] tokenEmbeddings;

    public Generate(Config config, Tokenizer tokenizer, BaseTransformer transformer)
    {
        this.config = config;
        this.tokenizer = tokenizer;
        this.transformer = transformer;
        this.tokenEmbeddings = transformer.matrix(ParameterType.TOKEN_EMBEDDINGS);
    }

    /**
     * Transformer token processor
     * Implements the logic how the input tokens and the new and new generated tokens are passed to the transformer
     */
    public List<Integer> process(List<Integer> inputTokens, int startPos)
    {
        List<Integer> result = new ArrayList<>();
        int intputSize = inputTokens.size();

        // Process the input tokens (except the last)
        if (intputSize == 0)
        {
            // If the input is empty, use the END_OF_TEXT token as input
            inputTokens.add(config.getEndOfTextToken());
            intputSize = 1;
        }
        else
        {
            // Iterating over on the input tokens (excluding the last one) and processing these by the transformer
            // We are not interested in the output of the transformer, but the inner state will be stored
            for (int pos = 0; pos < intputSize - 1; pos++)
            {
                OUT.print("."); // Printing a dot to show there is a progress
                float[] embedding = tokenEmbeddings[inputTokens.get(pos)];
                transformer.execute(pos + startPos, embedding, false);
            }
        }

        // Process the last input token and repeat it with the newly generated tokens
        int token = inputTokens.get(intputSize - 1);
        OUT.println(". "); // Printing something to show there is a progress

        // Use the transformer again and again to generate new tokens
        for (int pos = intputSize - 1; pos < config.getLengthLimit() + intputSize; pos++)
        {
            // Add the last input token or the previously generated new token as input
            float[] embedding = tokenEmbeddings[token];
            float[] hiddenState = transformer.execute(pos + startPos, embedding, true);

            token = determineOutputToken(hiddenState);
            result.add(token);

            // Exit if the END_OF_TEXT token was chosen or the maximum length is reached
            if (token == config.getEndOfTextToken()) break;

            // Exit if we reached the context size
            if (intputSize + result.size() + startPos >= config.getContextSize()) break;
        }

        return result;
    }

    private int determineOutputToken(float[] hiddenState)
    {
        // Multiply (dot product) the output with all token embeddings.
        // It will give a higher value if the output is more similar to the token embedding
        float[] logits = UTIL.mulVectorByTransposedMatrix(hiddenState, tokenEmbeddings);

        // Sort (higher to lower) the result of the dot products, retaining the order (index) of the related token
        List<IndexedValue> orderedLogits = UTIL.reverseAndFilter(logits, config.getTopK());

        // Convert the logits to probabilities
        float[] probabilities = TransformerUtil.softmax(orderedLogits);

        // Pick one token randomly, using a weighted random selection
        int index = weightedRandomPick(probabilities);

        // Lookup the token id
        int selectedTokenId = orderedLogits.get(index).getIndex();

        // Print the generated token - It isn't perfect, because some words or letters represented by multiple tokens
        OUT.print(tokenizer.decode(Collections.singletonList(selectedTokenId)));

        return selectedTokenId;
    }

    /**
     * Weighted random selection from list of probabilities
     */
    public static int weightedRandomPick(float[] probabilities)
    {
        float sum = 0;
        float[] cumulativeProbabilities = new float[probabilities.length];

        for (int i = 0; i < probabilities.length; i++)
        {
            sum = sum + probabilities[i] * 100;
            cumulativeProbabilities[i] = sum;
        }

        int random = (int)(Math.random() * sum);

        int index = 0;
        for (int i = 0; i < probabilities.length; i++)
        {
            if (random < cumulativeProbabilities[i]) break;

            index ++;
        }

        return index;
    }

    public void clear()
    {
        transformer.clear();
    }
}
