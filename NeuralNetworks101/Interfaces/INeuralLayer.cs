using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks101
{
    public interface INeuralLayer : IList<INeuron>
    {
        void Pulse(INeuralNet net);
        void ApplyLearning(INeuralNet net);
        void InitializeLearning(INeuralNet net);
    }
}
