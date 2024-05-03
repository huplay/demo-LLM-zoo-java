package huplay.demo.tokenizer.sentencePiece.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TokenizerConfig
{
    @JsonProperty
    public String version;

    @JsonProperty("added_tokens")
    public List<AddedTokenConfig> addedTokens;

    //@JsonProperty
    //public SentencePieceNormalizerConfig normalizer;

    //@JsonProperty("post_processor")
    //public SentencePiecePostProcessorConfig postProcessor;

    //@JsonProperty
    //public SentencePieceDecoderConfig decoder;

    @JsonProperty
    public ModelConfig model;

    @Override
    public String toString()
    {
        return "Config{" +
                "version='" + version + '\'' +
                ", addedTokens=" + addedTokens +
                ", model=" + model +
                '}';
    }
}
