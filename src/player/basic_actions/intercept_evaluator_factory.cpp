// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 3 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "intercept_evaluator_factory.h"

#include "intercept_evaluator_sample.h"

#include <rcsc/common/logger.h>

/*-------------------------------------------------------------------*/
InterceptEvaluatorFactory::InterceptEvaluatorFactory()
{
    init();
}


/*-------------------------------------------------------------------*/
InterceptEvaluatorFactory &
InterceptEvaluatorFactory::instance()
{
    static InterceptEvaluatorFactory s_instance;
    return s_instance;
}


/*-------------------------------------------------------------------*/
bool
InterceptEvaluatorFactory::init()
{
    static bool s_init = false;
    if ( s_init )
    {
        return true;
    }

    M_evaluators["Sample"] = InterceptEvaluator::Ptr( new InterceptEvaluatorSample() );

    s_init = true;
    return true;
}

/*-------------------------------------------------------------------*/
InterceptEvaluator::Ptr
InterceptEvaluatorFactory::create( const rcsc::WorldModel & /*wm*/ ) const
{
    InterceptEvaluator::Ptr ptr;

    std::string evaluator_name = "Default";

    if ( evaluator_name.empty()
         || evaluator_name == "Default" )
    {
        ptr = InterceptEvaluator::Ptr( new InterceptEvaluatorSample() );
    }
    else
    {
        decltype( M_evaluators )::const_iterator it = M_evaluators.find( evaluator_name );
        if ( it != M_evaluators.end() )
        {
            return it->second;
        }
    }

    return ptr;

}
