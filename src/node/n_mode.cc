/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_mode.h"

namespace FT{
    	
    NodeMode::NodeMode()
    {
        name = "mode";
	    otype = 'f';
	    arity['z'] = 1;
	    complexity = 1;
    }

#ifndef USE_CUDA
    /// Evaluates the node and updates the stack states. 
    void NodeMode::evaluate(const Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x]).mean();
            
        stack.z.pop();

        stack.push<double>(tmp);
        
    }
#else
    void NodeMode::evaluate(const Data& data, Stacks& stack)
    {
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            stack.f.row(stack.idx['f']) = stack.z.top().first[x].mean();
            
        stack.z.pop();

        
   }
#endif

    /// Evaluates the node symbolically
    void NodeMode::eval_eqn(Stacks& stack)
    {
        stack.push<double>("mean(" + stack.zs.pop() + ")");
    }
    
    NodeMode* NodeMode::clone_impl() const { return new NodeMode(*this); }

    NodeMode* NodeMode::rnd_clone_impl() const { return new NodeMode(); }  
}
