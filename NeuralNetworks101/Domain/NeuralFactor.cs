﻿using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks101.Domain
{
    public class NeuralFactor
    {
        #region Constructors  
        public NeuralFactor(double weight)
        {
            m_weight = weight;
            m_delta = 0;
        }
        #endregion

        #region Member Variables  
        private double m_weight;
        private double m_delta;
        #endregion

        #region Properties  
        public double Weight
        {
            get { return m_weight; }
            set { m_weight = value; }
        }
        public double Delta
        {
            get { return m_delta; }
            set { m_delta = value; }
        }
        #endregion

        #region Methods  
        public void ApplyDelta()
        {
            m_weight += m_delta;
            m_delta = 0;
        }
        #endregion
    }
}
