package huplay.demo.util;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Util extends AbstractUtil
{
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_MAX;

    @Override
    public String getUtilName()
    {
        return "Java Vector API (" + SPECIES.vectorBitSize() + " bit)";
    }

    @Override
    public Vector addVectors(Vector vector1, Vector vector2)
    {
        float[] result = new float[vector1.size()];

        for (int i = 0; i < vector1.size(); i += SPECIES.length())
        {
            VectorMask<Float> mask = SPECIES.indexInRange(i, vector1.size());
            FloatVector first = FloatVector.fromArray(SPECIES, vector1.getFloat32Values(), i, mask);
            FloatVector second = FloatVector.fromArray(SPECIES, vector2.getFloat32Values(), i, mask);
            first.add(second).intoArray(result, i, mask);
        }

        return new Vector(vector1.getFloatType(), result);
    }

    @Override
    public float dotProduct(Vector vector1, Vector vector2)
    {
        var upperBound = SPECIES.loopBound(vector1.size());
        var sum = FloatVector.zero(SPECIES);

        var i = 0;
        for (; i < upperBound; i += SPECIES.length())
        {
            var va = FloatVector.fromArray(SPECIES, vector1.getFloat32Values(), i);
            var vb = FloatVector.fromArray(SPECIES, vector2.getFloat32Values(), i);
            sum = va.fma(vb, sum);
        }

        var result = sum.reduceLanes(VectorOperators.ADD);

        // counter "i" has an incremented value from the previous loop
        for (; i < vector1.size(); i++)
        {
            result += vector1.get(i) * vector2.get(i);
        }

        return result;
    }

    @Override
    public Vector mulVectorByScalar(Vector vector, float scalar)
    {
        float[] result = new float[vector.size()];

        for (int i = 0; i < vector.size(); i += SPECIES.length())
        {
            VectorMask<Float> mask = SPECIES.indexInRange(i, vector.size());
            FloatVector floatVector = FloatVector.fromArray(SPECIES, vector.getFloat32Values(), i, mask);
            floatVector.mul(scalar).intoArray(result, i, mask);
        }

        return new Vector(vector.getFloatType(), result);
    }

    @Override
    // TODO: Vector-api isn't used
    public Vector mulVectorByMatrix(Vector vector, Vector[] matrix)
    {
        Vector ret = new Vector(vector.getFloatType(), matrix[0].size());

        for (int col = 0; col < matrix[0].size(); col++)
        {
            float sum = 0;

            for (int i = 0; i < vector.size(); i++)
            {
                sum = sum + vector.get(i) * matrix[i].get(col);
            }

            ret.set(col, sum);
        }

        return ret;
    }

    @Override
    public Vector mulVectorByTransposedMatrix(Vector vector, Vector[] matrix)
    {
        Vector ret = new Vector(vector.getFloatType(), matrix.length);

        for (int col = 0; col < matrix.length; col++)
        {
            ret.set(col, dotProduct(vector, matrix[col]));
        }

        return ret;
    }

    @Override
    // TODO: Vector-api isn't used
    public Vector[] splitVector(Vector vector, int count)
    {
        int size = vector.size() / count;
        Vector[] ret = Vector.newVectorArray(vector.getFloatType(), count, size);

        int segment = 0;
        int col = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            ret[segment].set(col, value);

            if (col == size - 1)
            {
                col = 0;
                segment++;
            }
            else col++;
        }

        return ret;
    }

    @Override
    // TODO: Vector-api isn't used
    public Vector flattenMatrix(Vector[] matrix)
    {
        Vector ret = new Vector(matrix[0].getFloatType(), matrix.length * matrix[0].size());

        int i = 0;

        for (Vector row : matrix)
        {
            for (int j = 0; j < row.size(); j++)
            {
                float value = row.get(j);
                ret.set(i, value);
                i++;
            }
        }

        return ret;
    }

    @Override
    // TODO: Vector-api isn't used
    public float average(Vector vector)
    {
        double sum = 0;

        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            sum = sum + value;
        }

        return (float) sum / vector.size();
    }
}