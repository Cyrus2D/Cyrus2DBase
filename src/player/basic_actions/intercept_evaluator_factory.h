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

#ifndef INTERCEPT_EVALUATOR_FACTORY_H
#define INTERCEPT_EVALUATOR_FACTORY_H

#include "intercept_evaluator.h"

#include <unordered_map>

namespace rcsc {
class WorldModel;
}

/*!
  \class InterceptEvaluatorFactory
  \breif abstract factory
*/
class InterceptEvaluatorFactory {
protected:
    InterceptEvaluatorFactory();

public:

    virtual
    ~InterceptevaluatorFactory()
      { }

    virtual
    InterceptEvaluator::Ptr create( const rcsc::WorldModel & wm ) const = 0;

};

#endif
