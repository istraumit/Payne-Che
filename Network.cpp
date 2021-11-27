#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>


class StreamReader{
private:
    std::ifstream* st;
union {
  float f;
  u_char b[4];
} uf;

union {
  int f;
  u_char b[4];
} u;
char * buf = new char[4];

public:
    StreamReader(std::ifstream* stream)
    {
        st = stream;
    }

    int read_int()
    {
        st->read(buf, 4);
        u.b[0] = buf[0];
        u.b[1] = buf[1];
        u.b[2] = buf[2];
        u.b[3] = buf[3];
        return u.f;
    }

    float read_float()
    {
        st->read(buf, 4);
        uf.b[0] = buf[0];
        uf.b[1] = buf[1];
        uf.b[2] = buf[2];
        uf.b[3] = buf[3];
        return uf.f;
    }
};
//-------------------------------------------------------------------
class Vector{
private:
    int _d0;
public:
    float * data;
    Vector(int d0)
    {
        _d0 = d0;
        data = (float *)malloc(sizeof(float)*d0);
    }

    void read_from_stream(StreamReader sr)
    {
        for(int i=0; i<_d0; i++)
            data[i] = sr.read_float();
    }

    void add_inplace(Vector v)
    {
        int n = v.get_d0();
        if(n != _d0)
            throw std::runtime_error("Incompatible vector lengths "+std::to_string(n)+" and "+std::to_string(_d0));
        for(int i=0; i<n; i++)
            data[i] += v.data[i];
    }

    int get_d0(){return _d0;}
};
//------------------------------------------------------------
class Matrix{
private:
    int _d0, _d1;
public:
    float ** data;
    Matrix(int d0, int d1)
    {
        _d0 = d0;
        _d1 = d1;
        data = (float **)malloc(sizeof(float*)*d0);
        for(int i=0; i<d0; i++)
            data[i] = (float*)malloc(sizeof(float)*d1);
    }

    void read_from_stream(StreamReader sr)
    {
        for(int i=0; i<_d0; i++)
        for(int j=0; j<_d1; j++)
            data[i][j] = sr.read_float();
    }

    Vector mul_vector(Vector v)
    {
        int vd0 = v.get_d0();
        if(vd0 != _d1)
            throw std::runtime_error("Incompatible matrix and vector dimensions");
        Vector V = Vector(_d0);

        for (int i = 0; i < _d0; i++)
        {
            V.data[i] = 0;
            for (int k = 0; k < _d1; k++)
                V.data[i] += data[i][k] * v.data[k];
        }
        return V;
    }
};
//-------------------------------------------------------------------

class Network{
private:

    Matrix* w_array_0;
    Matrix* w_array_1;
    Matrix* w_array_2;
    Vector* b_array_0;
    Vector* b_array_1;
    Vector* b_array_2;

    float leaky_relu(float z)
    {
        if(z<0) return z;
        return 0.01*z;
    }

    Vector act_func(Vector vin)
    {
        int n = vin.get_d0();
        auto vout = Vector(n);
        for(int i=0; i<n; i++)
            vin.data[i] = leaky_relu( vout.data[i] );
        return vout;
    }

public:
    Network(std::string path)
    {
        std::ifstream is (path, std::ifstream::binary);
        auto sr = StreamReader(&is);

        int n0 = sr.read_int();
        int n1 = sr.read_int();
        int n2 = sr.read_int();
        printf("n0 = %i\n", n0);
        printf("n1 = %i\n", n1);
        printf("n2 = %i\n", n2);

        w_array_0 = new Matrix(n1, n0);
        w_array_1 = new Matrix(n1, n1);
        w_array_2 = new Matrix(n2, n1);

        b_array_0 = new Vector(n1);
        b_array_1 = new Vector(n1);
        b_array_2 = new Vector(n2);

        w_array_0->read_from_stream(sr);
        w_array_1->read_from_stream(sr);
        w_array_2->read_from_stream(sr);

        b_array_0->read_from_stream(sr);
        b_array_1->read_from_stream(sr);
        b_array_2->read_from_stream(sr);
    }

    Vector get_output(Vector input)
    {
        Vector inside = w_array_0->mul_vector(input);
        inside.add_inplace(*b_array_0);

        Vector outside = w_array_1->mul_vector(act_func(inside));
        outside.add_inplace(*b_array_1);

        Vector spectrum = w_array_2->mul_vector(act_func(outside));
        spectrum.add_inplace(*b_array_2);

        return spectrum;
    }
};



int main()
{
auto NN = Network("z_bin_test");
printf("Network loaded\n");
auto vin = Vector(5);
for(int i=0; i<4000; i++)
{
auto out = NN.get_output(vin);
printf("%i\n", i);
}
return 0;
}



