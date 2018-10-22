/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_slope.h"
#include "../../../util/utils.h"    	

namespace FT{

    namespace Pop{
        namespace Op{
            NodeSlope::NodeSlope()
            {
                name = "slope";
	            otype = 'f';
	            arity['z'] = 1;
	            complexity = 4;
            }

            /// Evaluates the node and updates the stack states. 
            void NodeSlope::evaluate(const Data& data, Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                
                for(int x = 0; x < stack.z.top().first.size(); x++)                    
                    tmp(x) = slope(limited(stack.z.top().first[x]), limited(stack.z.top().second[x]));
                    
                stack.z.pop();

                stack.push<double>(tmp);
                
            }

            /// Evaluates the node symbolically
            void NodeSlope::eval_eqn(Stacks& stack)
            {
                stack.push<double>("slope(" + stack.zs.pop() + ")");
            }
            
            double NodeSlope::slope(const ArrayXd& x, const ArrayXd& y)
            {
                double varx = variance(x);
                if (varx > NEAR_ZERO)
                    return covariance(x, y)/varx;
                else
                    return 0;
            }

            NodeSlope* NodeSlope::clone_impl() const { return new NodeSlope(*this); }

            NodeSlope* NodeSlope::rnd_clone_impl() const { return new NodeSlope(); } 
        }
    }
}
