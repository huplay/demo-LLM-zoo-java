package huplay.demo.config;

import java.util.List;

public class ParameterDescriptor
{
    private final String fileName;
    private final String id;
    private final long dataOffset;
    private final String format;
    private final DataType dataType;
    private final List<Long> shape;
    private final long startOffset;
    private final long endOffset;

    public ParameterDescriptor(String fileName, String id, long dataOffset, String format, DataType dataType,
                               List<Long> shape, long startOffset, long endOffset)
    {
        this.fileName = fileName;
        this.id = id;
        this.dataOffset = dataOffset;
        this.format = format;
        this.dataType = dataType;
        this.shape = shape;
        this.startOffset = startOffset;
        this.endOffset = endOffset;
    }

    public String getFileName()
    {
        return fileName;
    }

    public String getId()
    {
        return id;
    }

    public long getDataOffset()
    {
        return dataOffset;
    }

    public String getFormat()
    {
        return format;
    }

    public DataType getDataType()
    {
        return dataType;
    }

    public List<Long> getShape()
    {
        return shape;
    }

    public long getStartOffset()
    {
        return startOffset;
    }

    public long getEndOffset()
    {
        return endOffset;
    }

    public long getSizeInBytes()
    {
        return endOffset - startOffset;
    }

    @Override
    public String toString()
    {
        return "ParameterDescriptor{" +
                "fileName='" + fileName + '\'' +
                ", id='" + id + '\'' +
                ", dataOffset=" + dataOffset +
                ", format='" + format + '\'' +
                ", dataType=" + dataType +
                ", shape=" + shape +
                ", startOffset=" + startOffset +
                ", endOffset=" + endOffset +
                '}';
    }
}
