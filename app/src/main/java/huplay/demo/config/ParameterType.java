package huplay.demo.config;

public enum ParameterType
{
    // Input
    TOKEN_EMBEDDINGS,
    POSITION_EMBEDDINGS,

    INPUT_NORM_WEIGHT,
    INPUT_NORM_BIAS,


    // Attention block
    ATT_QUERY_KEY_VALUE_WEIGHT,
    ATT_QUERY_KEY_VALUE_BIAS,

    ATT_QUERY_WEIGHT,
    ATT_QUERY_BIAS,

    ATT_KEY_WEIGHT,
    ATT_KEY_BIAS,

    ATT_VALUE_WEIGHT,
    ATT_VALUE_BIAS,

    ATT_PROJ_WEIGHT,
    ATT_PROJ_BIAS,

    ATT_NORM_WEIGHT,
    ATT_NORM_BIAS,


    // Feed-forward block
    MLP_1_WEIGHT,
    MLP_1_BIAS,

    MLP_2_WEIGHT,
    MLP_2_BIAS,

    MLP_3_WEIGHT,
    MLP_3_BIAS,

    MLP_NORM_WEIGHT,
    MLP_NORM_BIAS,


    // Output
    OUTPUT_NORM_WEIGHT,
    OUTPUT_NORM_BIAS
}
