package huplay.demo.transformer;

import huplay.demo.IdentifiedException;
import huplay.demo.config.Config;
import huplay.demo.transformer._2018_01_google_transformer.Transformer;
import huplay.demo.transformer._2021_03_eleuther_gptneo.GPTNeo;
import huplay.demo.transformer._2022_05_big_science_bloom.Bloom;
import huplay.demo.transformer._2023_02_meta_llama.Llama;
import huplay.demo.transformer._2018_06_openai_gpt1.GPT1;
import huplay.demo.transformer._2019_02_openai_gpt2.GPT2;

public enum TransformerType
{
    ORIGINAL_TRANSFORMER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    ELEUTHERAI_NEO,
    BIG_SCIENCE_BLOOM,
    META_LLAMA;

    public static BaseTransformer getTransformer(Config config)
    {
        String type = config.getTransformerType();
        if (type == null)
        {
            throw new IdentifiedException("Transformer type isn't specified");
        }

        type = type.toUpperCase();
        TransformerType transformerType = TransformerType.valueOf(type);
        switch (transformerType)
        {
            case ORIGINAL_TRANSFORMER: return new Transformer(config);
            case OPENAI_GPT_1: return new GPT1(config);
            case OPENAI_GPT_2: return new GPT2(config);
            case ELEUTHERAI_NEO: return new GPTNeo(config);
            case BIG_SCIENCE_BLOOM: return new Bloom(config);
            case META_LLAMA: return new Llama(config);
            default:
                throw new IdentifiedException("Unknown transformer type: " + type);
        }
    }
}
