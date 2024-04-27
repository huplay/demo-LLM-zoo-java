package huplay.demo.tokenizer;

import huplay.demo.config.Config;

import java.util.List;

public interface Tokenizer
{
    /**
     * Convert text to list of tokens
     */
    List<Integer> encode(String text);

    /**
     * Convert list of tokens to text
     */
    String decode(List<Integer> tokens);

    static Tokenizer getInstance(Config config)
    {
        String path = "tokenizerConfig/" + config.getTokenizerConfig();

        switch (config.getTokenizer())
        {
            case "GPT-1": return new GPT1Tokenizer(path);
            case "GPT-2": return new GPT2Tokenizer(path);
            case "SentencePiece": return new SentencePieceTokenizer(path, config.getTokenCount());
        }

        throw new RuntimeException("Unknown tokenizer: " + config.getTokenizer());
    }
}
