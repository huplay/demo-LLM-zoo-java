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
 * Tokenizer which is similar to OpenAI's GPT-1 tokenizer (Not fully compatible, but for most cases should work)
 * https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/openai/tokenization_openai.py
 *
 * @author Hunor Szegi
 */
public class GPT1Tokenizer implements Tokenizer
{
    private final Map<Integer, Character> charDecoding = new HashMap<>(478);

    private final Map<String, Integer> tokenEncoding;
    private final Map<Integer, String> tokenDecoding = new HashMap<>(40000);

    private final Map<Pair, Integer> merges;

    private final Pattern pattern =
            Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public GPT1Tokenizer(Config config)
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
            throw new IdentifiedException("GPT-1 tokenizer vocabulary reading error.", e);
        }

        File mergesFile = config.getModelConfig().findFile("merges.txt");
        if (!mergesFile.exists() || !mergesFile.isFile())
        {
            throw new IdentifiedException("GPT-1 tokenizer merges file is missing. (" + mergesFile.getName() + ")");
        }

        merges = FileReader.readMergesFile(mergesFile, true);
    }

    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        text = text.replace("—", "-");
        text = text.replace("–", "-");
        text = text.replace("―", "-");
        text = text.replace("…", "...");
        text = text.replace("´", "'");
        //text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
        //text = re.sub(r"\s*\n\s*", " \n ", text)
        //text = re.sub(r"[^\S\n]+", " ", text)
        text = text.toLowerCase().trim();

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
            String word = tokenDecoding.get(token);

            if (word != null)
            {
                if (word.endsWith("</w>"))
                {
                    word = word.substring(0, word.length() - 4) + " ";
                }

                textBuilder.append(word);
            }
        }
        return textBuilder.toString();
    }

    private void addCharRange(int pos, int firstChar, int lastChar)
    {
        for (int i = firstChar; i <= lastChar; i++)
        {
            charDecoding.put(pos, (char) i);
            pos++;
        }
    }
}
