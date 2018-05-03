/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#define NEAR_ZERO 0.00001  // Added in as placeholder, ask Bill what number this should be

#include "nodeDx.h"

namespace FT{
	class NodeDivide : public NodeDx
    {
    	public:
    	  	
    		NodeDivide()
    		{
    			name = "/";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;

                for (int i = 0; i < arity['f']; i++) {
                    W.push_back(1);
                }
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                ArrayXd x1 = stack.f.pop();
                ArrayXd x2 = stack.f.pop();
                // safe division returns x1/x2 if x2 != 0, and MAX_DBL otherwise               
                stack.f.push( (abs(x2) > NEAR_ZERO ).select((this->W[1] * x1) / (this->W[0] * x2), 
                                                            1.0) ); 
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "/" + stack.fs.pop() + ")");            	
            }

            // Might want to check derivative orderings for other 2 arg nodes
            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                ArrayXd x1 = stack_f[stack_f.size() - 2];
                ArrayXd x2 = stack_f[stack_f.size() - 1];
                switch (loc) {
                    case 3:
                        return limited(x1/(this->W[0] * x2));
                    case 2:
                        return limited(-this->W[1] * x1/(x2 * pow(this->W[0], 2)));
                    case 1:
                    {
                        return limited(this->W[1]/(this->W[0] * x2));
                        // ArrayXd num = -this->W[1] * x1;
                        // ArrayXd denom = limited(this->W[0] * pow(x2, 2));
                        // ArrayXd val = num/denom;
                        // return val;
                    }
                    case 0: // with respect to first element off stack (x2)
                    default:
                        ArrayXd num = -this->W[1] * x1;
                        ArrayXd denom = limited(this->W[0] * pow(x2, 2));
                        ArrayXd val = num/denom;
                        return val;
                        // return limited(this->W[1]/(this->W[0] * x2));
                } 
            }
            
        protected:
            NodeDivide* clone_impl() const override { return new NodeDivide(*this); };  
    };
}	

#endif
