using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetworks101.Domain;
using NeuralNetworks101.Interfaces;

namespace NeuralNetworks101
{
    public interface INeuron : INeuronSignal, INeuronReceptor
    {
        void Pulse(INeuralLayer layer);
        void ApplyLearning(INeuralLayer layer, ref double learningRate);
        void InitializeLearning(INeuralLayer layer);

        NeuralFactor Bias { get; set; }

        double Error { get; set; }
        double LastError { get; set; }
    }
}
