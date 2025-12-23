import os
import sys
from typing import Optional, Dict, List

# Add the project root to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.user_roles import email_to_role_mapping, role_page_access, default_role
except ImportError:
    # Fallback if config file is not available
    email_to_role_mapping = {}
    role_page_access = {}
    default_role = None


def get_user_role(email: str) -> Optional[str]:
    """
    Get the role for a given email address.
    
    Args:
        email: User's email address
        
    Returns:
        Role name if found, None if not found
    """
    if not email:
        return default_role
    
    # Check direct email mapping
    role = email_to_role_mapping.get(email.lower())
    if role:
        return role
    
    # Check domain-based access for skillcatapp.com
    email_lower = email.lower()
    if email_lower.endswith("@skillcatapp.com"):
        return "Basic User"
    
    # If no direct mapping found, return default role
    return default_role


def get_user_pages(role: str) -> List[str]:
    """
    Get the list of pages a user with the given role can access.
    
    Args:
        role: User's role
        
    Returns:
        List of page names the user can access
    """
    if not role:
        return []
    
    return role_page_access.get(role, [])


def is_user_authorized(email: str) -> bool:
    """
    Check if a user with the given email is authorized to access the application.
    
    Args:
        email: User's email address
        
    Returns:
        True if user is authorized, False otherwise
    """
    role = get_user_role(email)
    return role is not None and role != default_role


def get_user_info(email: str) -> Dict[str, any]:
    """
    Get comprehensive user information including role and page access.
    
    Args:
        email: User's email address
        
    Returns:
        Dictionary with user role and accessible pages
    """
    role = get_user_role(email)
    pages = get_user_pages(role) if role else []
    
    return {
        "email": email,
        "role": role,
        "pages": pages,
        "is_authorized": is_user_authorized(email)
    }
