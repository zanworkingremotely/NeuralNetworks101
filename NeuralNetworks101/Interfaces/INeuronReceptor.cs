using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetworks101.Domain;

namespace NeuralNetworks101.Interfaces
{
    public interface INeuronReceptor
    {
        Dictionary<INeuronSignal, NeuralFactor> Input { get; }
    }
}
