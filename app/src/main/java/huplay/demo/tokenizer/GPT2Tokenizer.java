package huplay.demo.tokenizer;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer which fully compatible with the OpenAI GPT-2 and GPT-3 tokenizer (used for other models as well)
 *
 * @author Hunor Szegi
 */
public class GPT2Tokenizer implements Tokenizer
{
    private final Map<Character, Byte> charEncoding = new HashMap<>(256);
    private final Map<Integer, Character> charDecoding = new HashMap<>(256);

    private final Map<String, Integer> tokenEncoding;
    private final Map<Integer, String> tokenDecoding = new HashMap<>(50257);

    private final Map<Pair, Integer> merges;

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public GPT2Tokenizer(Config config)
    {
        addCharRange(0, 256, 288);
        addCharRange(33, 33, 126);
        addCharRange(127, 289, 322);
        addCharRange(161, 161, 172);
        addCharRange(173, 323, 323);
        addCharRange(174, 174, 255);

        File vocabularyFile = config.getModelConfig().findFile("vocab.json");
        if (!vocabularyFile.exists() || !vocabularyFile.isFile())
        {
            throw new IdentifiedException("GPT-1 tokenizer vocabulary file is missing. (" + vocabularyFile.getName() + ")");
        }

        try
        {
            TypeReference<Map<String, Integer>> typeRef = new TypeReference<>() {};
            tokenEncoding = new ObjectMapper().readValue(vocabularyFile, typeRef);

            tokenEncoding.forEach((key, value) -> tokenDecoding.put(value, key));
        }
        catch (IOException e)
        {
            throw new IdentifiedException("GPT-2 tokenizer vocabulary reading error.", e);
        }

        File mergesFile = config.getModelConfig().findFile("merges.txt");
        if (!mergesFile.exists() || !mergesFile.isFile())
        {
            throw new IdentifiedException("GPT-2 tokenizer merges file is missing. (" + mergesFile.getName() + ")");
        }

        merges = FileReader.readMergesFile(mergesFile, true);
    }

    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        List<Integer> result = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);
        List<String> unicodeText = new ArrayList<>();

        while (matcher.find())
        {
            StringBuilder match = new StringBuilder();

            ByteBuffer buffer = StandardCharsets.UTF_8.encode(matcher.group());
            while (buffer.hasRemaining())
            {
                int value = buffer.get();
                if (value < 0) value = value & 0xff;
                match.append(charDecoding.get(value));
            }

            unicodeText.add(match.toString());
        }

        for (String word : unicodeText)
        {
            for (String token : BytePairEncoding.encode(word, merges).split(" "))
            {
                Integer value = tokenEncoding.get(token);
                if (value != null)
                {
                    result.add(value);
                }
            }
        }

        return result;
    }

    public String decode(List<Integer> tokens)
    {
        StringBuilder textBuilder = new StringBuilder();
        for (int token : tokens)
        {
            textBuilder.append(tokenDecoding.get(token));
        }
        String text = textBuilder.toString();

        byte[] bytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
        {
            bytes[i] = charEncoding.get(text.charAt(i));
        }

        return new String(bytes, StandardCharsets.UTF_8);
    }

    private void addCharRange(int pos, int firstChar, int lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charEncoding.put((char) i, (byte)pos);
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }
}
