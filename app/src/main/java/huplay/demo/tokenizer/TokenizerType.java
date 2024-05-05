package huplay.demo.tokenizer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;
import huplay.demo.tokenizer.gpt.GPT1Tokenizer;
import huplay.demo.tokenizer.gpt.GPT2Tokenizer;
import huplay.demo.tokenizer.sentencePiece.SentencePieceTokenizer;

public enum TokenizerType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    SENTENCE_PIECE,
    TIKTOKEN;

    public static Tokenizer getTokenizer(Config config)
    {
        String type = config.getTokenizerType();
        if (type == null)
        {
            throw new IdentifiedException("Tokenizer type isn't specified");
        }

        type = type.toUpperCase();
        String variant = "";
        if (type.contains("/"))
        {
            int index = type.indexOf("/");
            variant = type.substring(index + 1);
            type = type.substring(0, index);
        }

        TokenizerType tokenizerType = TokenizerType.valueOf(type);

        switch (tokenizerType)
        {
            case OPENAI_GPT_1: return new GPT1Tokenizer(config);
            case OPENAI_GPT_2: return new GPT2Tokenizer(config);
            case SENTENCE_PIECE: return new SentencePieceTokenizer(config, variant);
            case TIKTOKEN:
            default:
                throw new IdentifiedException("Unknown tokenizer type: " + type);
        }
    }
}
