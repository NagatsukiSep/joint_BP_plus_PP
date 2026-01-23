#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

struct Inputs
{
    double C_i = 0.0;
    double C_c = 0.0;
    std::vector<double> p;
};

static void usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " input_file\n";
}

static bool read_input_file(const std::string &path, Inputs &out)
{
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    bool have_header = false;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;
        if (!have_header)
        {
            char *end1 = nullptr;
            char *end2 = nullptr;
            const double c_i = std::strtod(line.c_str(), &end1);
            if (end1 == line.c_str())
                continue;
            const double c_c = std::strtod(end1, &end2);
            if (end2 == end1)
                continue;
            out.C_i = c_i;
            out.C_c = c_c;
            have_header = true;
            continue;
        }
        char *endn = nullptr;
        char *endp = nullptr;
        const long n = std::strtol(line.c_str(), &endn, 10);
        if (endn == line.c_str())
            continue;
        const double p = std::strtod(endn, &endp);
        if (endp == endn)
            continue;
        if (n < 0)
            continue;
        if (static_cast<size_t>(n) >= out.p.size())
        {
            out.p.resize(static_cast<size_t>(n) + 1, 0.0);
        }
        out.p[static_cast<size_t>(n)] = p;
    }
    return have_header;
}

// TODO: Replace with your real objective function.
static double compute_objective(int W, int I, const Inputs &in)
{
    // Placeholder: simple weighted sum using p[n].
    double acc = 0.0;
    for (size_t n = 0; n < in.p.size(); ++n)
    {
        if (n > W)
        {
            int k = (n - W) / I + ((n - W) % I != 0 ? 1 : 0);
            acc += in.p[n] * ((in.C_c + in.C_i * I) * k + in.C_i * W - in.C_i * I);
        }
        else
        {
            acc += in.p[n] * (in.C_i * W + in.C_c * I);
        }
    }
    return acc;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        usage(argv[0]);
        return 1;
    }

    Inputs in;
    if (!read_input_file(argv[1], in))
    {
        std::cerr << "Failed to read input file: " << argv[1] << "\n";
        return 1;
    }
    if (in.p.empty())
    {
        std::cerr << "p data is empty or invalid: " << argv[1] << "\n";
        return 1;
    }

    int n_up = -1;
    for (size_t n = 0; n < in.p.size(); ++n)
    {
        if (in.p[n] != 0.0)
        {
            n_up = static_cast<int>(n);
        }
    }
    if (n_up < 0)
    {
        std::cerr << "All p[n] are zero; no valid n_up.\n";
        return 1;
    }

    const int W_min = 0;
    const int W_max = n_up;

    struct Result {
        int W;
        int I;
        double obj;
    };
    std::vector<Result> top;
    top.reserve(5);

    for (int W = W_min; W <= W_max; ++W)
    {
        const int I_min = 1;
        const int I_max = n_up - W + 1; // smallest I where W + I > n_up
        for (int I = I_min; I <= I_max; ++I)
        {
            const double obj = compute_objective(W, I, in);
            top.push_back(Result{W, I, obj});
            std::sort(top.begin(), top.end(), [](const Result &a, const Result &b) {
                return a.obj < b.obj;
            });
            if (top.size() > 5) top.pop_back();
        }
    }

    std::cout << "top5 (best to worst)\n";
    for (size_t i = 0; i < top.size(); ++i)
    {
        std::cout << (i + 1) << ": W=" << top[i].W
                  << " I=" << top[i].I
                  << " obj=" << top[i].obj << "\n";
    }
    return 0;
}
