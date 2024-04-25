package huplay.demo.config;

/**
 * Holder of the app's input parameters
 */
public class Arguments
{
    private final String name;
    private final String configRoot;
    private final String modelRoot;
    private final int lengthLimit;
    private final int topK;
    private final boolean isCalculationOnly;

    public Arguments(String name, String configRoot, String modelRoot, int lengthLimit, int topK,
                     boolean isCalculationOnly)
    {
        this.name = name;
        this.configRoot = configRoot;
        this.modelRoot = modelRoot;
        this.lengthLimit = lengthLimit;
        this.topK = topK;
        this.isCalculationOnly = isCalculationOnly;
    }

    public String getName()
    {
        return name;
    }

    public String getConfigRoot()
    {
        return configRoot;
    }

    public String getModelRoot()
    {
        return modelRoot;
    }

    public int getLengthLimit()
    {
        return lengthLimit;
    }

    public int getTopK()
    {
        return topK;
    }

    public boolean isCalculationOnly()
    {
        return isCalculationOnly;
    }
}