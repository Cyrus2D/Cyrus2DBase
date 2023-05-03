// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa Akiyama

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

#ifndef INTERCEPT_EVALUATOR_H
#define INTERCEPT_EVALUATOR_H

#include <memory>
#include <functional>

#include <rcsc/factory.h>

namespace rcsc {
class Intercept;
class WorldModel;
}

/*!
  \class InterceptEvaluator
  \brief abstract evaluator function class
 */
class InterceptEvaluator {
public:
    using Ptr = std::shared_ptr< InterceptEvaluator >; //!< pointer type
    using Creator = std::function< Ptr() >; //!< creator function
    using Creators = rcss::Factory< Creator, std::string >;

    struct Evaluation {
        const rcsc::Intercept * info_;
        double value_;
    };

    static
    Creators & creators();

    static
    Ptr create( const std::string & name );

protected:

    /*!
      \brief protected constructor
     */
    InterceptEvaluator()
      { }

public:

    /*!
      \brief virtual destructor
     */
    virtual
    ~InterceptEvaluator()
      { }


    virtual
    bool isValid() const
      {
          return true;
      }

    /*!
      \brief pure virtual function.
      \return evaluation value
     */
    virtual
    double evaluate( const rcsc::WorldModel & wm,
                     const rcsc::Intercept & action,
                     const bool save_recovery ) = 0;
};

#endif
