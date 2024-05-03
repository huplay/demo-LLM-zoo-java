package huplay.demo.tokenizer.sentencePiece.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class AddedTokenConfig
{
    @JsonProperty
    public int id;

    @JsonProperty
    public String content;

    @JsonProperty("single_word")
    public boolean isSingleWord;

    @JsonProperty("lstrip")
    public boolean isLeftTrim;

    @JsonProperty("rstrip")
    public boolean isRightTrim;

    @JsonProperty("normalized")
    public boolean isNormalized;

    @JsonProperty("special")
    public boolean isSpecial;
}
