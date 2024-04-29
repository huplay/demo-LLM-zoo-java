package huplay.demo.config;

import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer._2017_google_transformer.Transformer;
import huplay.demo.transformer._2021_eleuther_gptneo.GPTNeo;
import huplay.demo.transformer._2022_huggingface_bloom.Bloom;
import huplay.demo.transformer._2023_meta_llama.Llama;
import huplay.demo.transformer._2018_openai_gpt1.GPT1;
import huplay.demo.transformer._2019_openai_gpt2.GPT2;

public enum TransformerType
{
    ORIGINAL_TRANSFORMER,
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    ELEUTHERAI_NEO,
    HUGGING_FACE_BLOOM,
    META_LLAMA;

    public BaseTransformer getTransformer(Config config)
    {
        switch (this)
        {
            case ORIGINAL_TRANSFORMER: return new Transformer(config);
            case OPENAI_GPT_1: return new GPT1(config);
            case OPENAI_GPT_2: return new GPT2(config);
            case ELEUTHERAI_NEO: return new GPTNeo(config);
            case HUGGING_FACE_BLOOM: return new Bloom(config);
            case META_LLAMA: return new Llama(config);
            default:
                throw new RuntimeException("Unknown transformer type: " + config.getTransformerType());
        }
    }
}
