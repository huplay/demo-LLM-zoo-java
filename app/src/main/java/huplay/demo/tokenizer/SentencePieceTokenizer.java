package huplay.demo.tokenizer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Java implementation of Google SentencePiece tokenizer (without training)
 * Original implementation: <a href="https://github.com/google/sentencepiece" />
 *
 * @author Hunor Szegi
 */
public class SentencePieceTokenizer implements Tokenizer
{
    private static final int START_OF_TEXT = 1;
    private static final String HEX_TOKEN_PREFIX = "<0x";
    private static final String HEX_TOKEN_SUFFIX = ">";

    protected String[] vocabulary;
    protected float[] vocabularyScores;
    private final Map<String, Integer> vocabularyIndex = new HashMap<>();

    public SentencePieceTokenizer(Config config)
    {
        init(config);

        // Create the vocabulary index, to quickly find the id of a token
        for (int i = 0; i < config.getTokenCount(); i++)
        {
            vocabularyIndex.put(vocabulary[i], i);
        }
    }

    protected void init(Config config)
    {
        // TODO: Read config from json

        this.vocabulary = new String[config.getTokenCount()];
        this.vocabularyScores = new float[config.getTokenCount()];

        File tokenizerFile = config.getModelConfig().findFile("tokenizer.json");
        if (!tokenizerFile.exists() || !tokenizerFile.isFile())
        {
            throw new IdentifiedException("SentencePiece tokenizer merges file is missing. (" + tokenizerFile.getName() + ")");
        }

        // Read the vocabulary from the binary config file
        Path configFilePath = Paths.get(tokenizerFile.getAbsolutePath());
        try (FileChannel channel = FileChannel.open(configFilePath, StandardOpenOption.READ))
        {
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // The first integer contains the maximum token length (we don't need this info)
            buffer.getInt();

            // Iterate over on all tokens
            for (int i = 0; i < config.getTokenCount(); i++)
            {
                // Read vocabulary score
                this.vocabularyScores[i] = buffer.getFloat();

                // Read token length
                int length = buffer.getInt();

                // Read token
                byte[] bytes = new byte[length];
                buffer.get(bytes);
                this.vocabulary[i] = new String(bytes, StandardCharsets.UTF_8);
            }
        }
        catch (Exception e)
        {
            throw new IdentifiedException("SentencePiece tokenizer vocabulary reading error.", e);
        }
    }

    private int getId(String text)
    {
        return vocabularyIndex.getOrDefault(text, -1);
    }

    @Override
    public String decode(List<Integer> tokens)
    {
        StringBuilder buffer = new StringBuilder();

        int prevToken = 0;
        for (int token : tokens)
        {
            buffer.append(decodeToken(prevToken, token));
            prevToken = token;
        }

        return buffer.toString();
    }

    private String decodeToken(int prevToken, Integer token)
    {
        String text = vocabulary[token];

        if (text == null)
        {
            return "<ERROR>";
        }

        if (text.length() == 6 && text.startsWith(HEX_TOKEN_PREFIX) && text.endsWith(HEX_TOKEN_SUFFIX))
        {
            // If it is a hex token (in the format of <0xNN>), convert the hex value (NN) to character
            String hex = text.substring(HEX_TOKEN_PREFIX.length(), HEX_TOKEN_PREFIX.length() + 2);
            text = Character.toString(Integer.parseInt(hex, 16));
        }
        else if (prevToken == START_OF_TEXT && text.charAt(0) == ' ')
        {
            // If the token is the first one with leading space, remove the space
            text = text.substring(1);
        }

        return text;
    }

    @Override
    public List<Integer> encode(String text)
    {
        if (text == null)
        {
            return Collections.singletonList(0);
        }
        else if (text.isEmpty())
        {
            return Collections.singletonList(START_OF_TEXT);
        }

        List<Integer> tokens = new ArrayList<>();

        tokens.add(START_OF_TEXT);

        // First encode the text as a sequence of individual unicode characters
        int codePoint;
        for (int i = 0; i < text.length(); i += Character.charCount(codePoint))
        {
            codePoint = text.codePointAt(i);
            String character = Character.toString(codePoint);

            int id = getId(character);

            if (id != -1)
            {
                // The character is in the vocabulary
                tokens.add(id);
            }
            else
            {
                // Rare unicode character, not in the vocabulary. Encode the character as a sequence of bytes
                // (All single-byte (0-255) and a lot of multibyte characters are in the vocabulary, but not all)
                // (The single-byte characters are stored after the <unk>, <s>, </s>, that's why the "+ 3")
                for (byte b : character.getBytes(StandardCharsets.UTF_8))
                {
                    tokens.add(Byte.toUnsignedInt(b) + 3);
                }
            }
        }

        // Merge pairs of encoded characters into a single token, using the scores to find the best merges
        while (true)
        {
            float bestScore = -1e10f;
            int bestId = -1;
            int bestIndex = -1;

            // Iterate over on the tokens (without the last one)
            // Find the best merge of token[i] and token[i + 1]
            for (int i = 0; i < tokens.size() - 1; i++)
            {
                String mergedText = vocabulary[tokens.get(i)] + vocabulary[tokens.get(i + 1)];
                int id = getId(mergedText);

                if (id != -1 && vocabularyScores[id] > bestScore)
                {
                    // If this text exists in the vocabulary, and has better score as the best, record it
                    bestScore = vocabularyScores[id];
                    bestId = id;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1)
            {
                // There's nothing to merge, we are ready
                break;
            }

            // Do the best merge
            tokens.set(bestIndex, bestId);
            tokens.remove(bestIndex + 1);
        }

        return tokens;
    }
}