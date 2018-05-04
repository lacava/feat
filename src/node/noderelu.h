/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_RELU
#define NODE_RELU

#include "nodeDx.h"

namespace FT{
	class NodeRelu : public NodeDx
    {
    	public:
    	  	
    		NodeRelu()
    		{
    			name = "relu";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(r.rnd_dbl());
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
             void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
                          Stacks& stack)
            {
                ArrayXd x = stack.f.pop();
                
                // ArrayXd res = (W[0] * x > 0).select((W[0] * x == 0).select(ArrayXd::Zero(x.size()), -1*ArrayXd::Ones(x.size()))); 
                ArrayXd res = x; // Need to replace with above line
                stack.f.push(res);
            }

            /// Evaluates the node symbolically
             void eval_eqn(Stacks& stack)
            {
                stack.fs.push("relu("+ stack.fs.pop() +")");         	
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {

                switch (loc) {
                    case 1: // d/dx1
                        return W[1]/(W[0] * stack_f[stack_f.size()-2]);
                    case 0: // d/dx0
                    default:
                        return -W[1] * stack_f[stack_f.size() - 1]/(W[0] * pow(stack_f[stack_f.size()], 2));
                } 
            }

            protected:
            NodeRelu* clone_impl() const override { return new NodeRelu(*this); };  
    };
}	

#endif