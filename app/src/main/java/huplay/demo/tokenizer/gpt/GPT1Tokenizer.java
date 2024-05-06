package huplay.demo.tokenizer.gpt;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;
import huplay.demo.tokenizer.Token;
import huplay.demo.tokenizer.Tokenizer;

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

        merges = MergesReader.readMergesFile(mergesFile, true);
    }

    @Override
    public List<Integer> encode(String text)
    {
        if (text == null) return Collections.singletonList(0);

        List<String> unicodeText = splitToUnicode(text);

        List<Integer> result = new ArrayList<>();

        for (String word : unicodeText)
        {
            for (String token : BytePairEncoding.encode(word, merges).split(" "))
            {
                Integer id = tokenEncoding.get(token);
                if (id != null)
                {
                    result.add(id);
                }
            }
        }

        return result;
    }

    private List<String> splitToUnicode(String text)
    {
        List<String> unicodeText = new ArrayList<>();

        text = text.replace("—", "-");
        text = text.replace("–", "-");
        text = text.replace("―", "-");
        text = text.replace("…", "...");
        text = text.replace("´", "'");
        //text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
        //text = re.sub(r"\s*\n\s*", " \n ", text)
        //text = re.sub(r"[^\S\n]+", " ", text)
        text = text.toLowerCase().trim();

        Matcher matcher = pattern.matcher(text);
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

        return unicodeText;
    }

    @Override
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

    @Override
    public List<Token> split(String text)
    {
        if (text == null || text.length() == 0) return Collections.emptyList();

        List<String> unicodeText = splitToUnicode(text);

        List<Token> result = new ArrayList<>();

        String prefix = "";
        for (String word : unicodeText)
        {
            for (String token : BytePairEncoding.encode(word, merges).split(" "))
            {
                Integer id = tokenEncoding.get(token);
                if (id != null)
                {
                    result.add(new Token(id, prefix + token));
                    prefix = "";
                }
                else
                {
                    prefix = " ";
                }
            }
        }

        return result;
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
