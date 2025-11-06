# Database Documentation - Comprehensive User Data Collection

**Database:** SQLite (`chatbot_data/users.db`)  
**Tables:** 6 comprehensive tables  
**Purpose:** Complete user analytics and system monitoring

---

## 📊 Database Schema

### Table 1: `users` - User Profiles

**Stores:** Complete user profile information

| Column | Type | Description |
|--------|------|-------------|
| `username` | TEXT PRIMARY KEY | Unique username |
| `password_hash` | TEXT | SHA-256 hashed password |
| `email` | TEXT | User email address |
| `full_name` | TEXT | Full name |
| `role` | TEXT | User role (student/admin) |
| `created_at` | TIMESTAMP | Account creation date |
| `last_login` | TIMESTAMP | Last login timestamp |
| `total_logins` | INTEGER | Total number of logins |
| `is_active` | BOOLEAN | Account active status |
| `preferences` | TEXT | User preferences (JSON) |

**Example Query:**
```sql
SELECT username, email, role, total_logins, last_login 
FROM users 
WHERE is_active = 1;
```

---

### Table 2: `conversations` - Chat History with Metadata

**Stores:** Every conversation with comprehensive metadata

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique conversation ID |
| `username` | TEXT | User who sent message |
| `session_id` | TEXT | Session identifier |
| `message` | TEXT | User's message |
| `response` | TEXT | AI's response |
| `model_used` | TEXT | Which AI model generated response |
| `tokens_generated` | INTEGER | Number of tokens in response |
| `response_time_ms` | INTEGER | Response time in milliseconds |
| `timestamp` | TIMESTAMP | When message was sent |
| `ip_address` | TEXT | User's IP address |
| `user_agent` | TEXT | Browser/client information |

**Example Query:**
```sql
SELECT username, model_used, tokens_generated, response_time_ms, timestamp
FROM conversations
WHERE username = 'student'
ORDER BY timestamp DESC
LIMIT 10;
```

---

### Table 3: `login_history` - Login Activity Tracking

**Stores:** All login attempts (successful and failed)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique log ID |
| `username` | TEXT | Username attempting login |
| `login_time` | TIMESTAMP | When login occurred |
| `logout_time` | TIMESTAMP | When user logged out |
| `session_duration_seconds` | INTEGER | How long user was logged in |
| `ip_address` | TEXT | IP address of login |
| `success` | BOOLEAN | Login successful (1) or failed (0) |

**Use Cases:**
- Security monitoring (failed login attempts)
- User activity patterns
- Session duration analysis
- Peak usage times

**Example Query:**
```sql
-- Find failed login attempts
SELECT username, login_time, ip_address
FROM login_history
WHERE success = 0
ORDER BY login_time DESC;
```

---

### Table 4: `usage_analytics` - Daily User Metrics

**Stores:** Aggregated daily statistics per user

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique record ID |
| `username` | TEXT | User |
| `date` | DATE | Date of activity |
| `total_messages` | INTEGER | Messages sent that day |
| `total_tokens` | INTEGER | Tokens generated that day |
| `avg_response_time_ms` | REAL | Average response time |
| `models_used` | TEXT | Which models used (JSON) |
| `session_count` | INTEGER | Number of sessions |

**Use Cases:**
- Track user engagement over time
- Identify most active users
- Measure daily usage patterns
- Token consumption tracking

**Example Query:**
```sql
-- Get user activity for last 7 days
SELECT date, username, total_messages, total_tokens
FROM usage_analytics
WHERE date >= DATE('now', '-7 days')
ORDER BY date DESC, total_messages DESC;
```

---

### Table 5: `model_stats` - AI Model Usage Statistics

**Stores:** Which models users prefer and their performance

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique record ID |
| `model_name` | TEXT | Name of AI model |
| `username` | TEXT | User |
| `usage_count` | INTEGER | How many times used |
| `total_tokens` | INTEGER | Total tokens generated |
| `avg_response_time_ms` | REAL | Average response time |
| `last_used` | TIMESTAMP | Last usage timestamp |

**Use Cases:**
- Identify most popular models
- Performance comparison between models
- User model preferences
- Resource allocation planning

**Example Query:**
```sql
-- Most popular models
SELECT model_name, SUM(usage_count) as total_uses, AVG(avg_response_time_ms) as avg_time
FROM model_stats
GROUP BY model_name
ORDER BY total_uses DESC;
```

---

### Table 6: `activity_log` - System Event Audit Trail

**Stores:** All system events for auditing

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique log ID |
| `event_type` | TEXT | Type of event (login, message, registration, admin_action) |
| `username` | TEXT | User who triggered event |
| `details` | TEXT | Event details |
| `timestamp` | TIMESTAMP | When event occurred |

**Event Types:**
- `login` - User logged in
- `logout` - User logged out
- `registration` - New user registered
- `message` - User sent message
- `admin_action` - Admin performed action
- `error` - System error occurred

**Example Query:**
```sql
-- Recent admin actions
SELECT timestamp, username, details
FROM activity_log
WHERE event_type = 'admin_action'
ORDER BY timestamp DESC;
```

---

## 📈 What Data is Collected

### On User Registration:
- ✅ Username (unique identifier)
- ✅ Password (hashed with SHA-256)
- ✅ Email address (optional)
- ✅ Full name (optional)
- ✅ Role assignment (student/admin)
- ✅ Registration timestamp
- ✅ Activity log entry

### On Every Login:
- ✅ Login timestamp
- ✅ Success/failure status
- ✅ IP address (local/remote)
- ✅ Login counter increment
- ✅ Last login update
- ✅ Activity log entry

### On Every Message:
- ✅ Full conversation content
- ✅ Session ID (track conversation flow)
- ✅ AI model used
- ✅ Tokens generated
- ✅ Response time (milliseconds)
- ✅ Timestamp
- ✅ IP address
- ✅ Daily analytics update
- ✅ Model usage stats update
- ✅ Activity log entry

### Automated Analytics:
- ✅ Daily message counts per user
- ✅ Token consumption tracking
- ✅ Model performance metrics
- ✅ User engagement patterns
- ✅ Peak usage times
- ✅ Average response times

---

## 🔍 Admin Dashboard Queries

### User Statistics:
```sql
-- Active users in last 7 days
SELECT username, COUNT(*) as messages, MAX(timestamp) as last_active
FROM conversations
WHERE timestamp >= DATE('now', '-7 days')
GROUP BY username
ORDER BY messages DESC;
```

### Performance Metrics:
```sql
-- Average response time by model
SELECT model_used, 
       AVG(response_time_ms) as avg_time,
       COUNT(*) as usage_count,
       AVG(tokens_generated) as avg_tokens
FROM conversations
WHERE model_used IS NOT NULL
GROUP BY model_used;
```

### User Engagement:
```sql
-- Messages per user
SELECT username, 
       COUNT(*) as total_messages,
       SUM(tokens_generated) as total_tokens,
       AVG(response_time_ms) as avg_response_time,
       MIN(timestamp) as first_message,
       MAX(timestamp) as last_message
FROM conversations
GROUP BY username
ORDER BY total_messages DESC;
```

### Login Patterns:
```sql
-- Login frequency by hour
SELECT strftime('%H', login_time) as hour, 
       COUNT(*) as login_count
FROM login_history
WHERE success = 1
GROUP BY hour
ORDER BY hour;
```

---

## 📊 Analytics Reports Available

### For Administrators:

**User Activity Report:**
- Total registered users
- Active users (last 24hr/7day/30day)
- User engagement scores
- Most active users
- Inactive user detection

**Usage Statistics:**
- Total conversations
- Messages per day/week/month
- Tokens consumed
- Average session length
- Peak usage times

**Model Performance:**
- Most popular models
- Response times by model
- Token efficiency
- User model preferences

**System Health:**
- Average response times
- Error rates
- GPU utilization
- Database size growth

### For Researchers:

**User Behavior Analysis:**
- Conversation patterns
- Question types
- Response satisfaction
- Model switching patterns

**Performance Optimization:**
- Identify slow queries
- Optimize popular models
- Resource allocation
- Capacity planning

---

## 🔒 Privacy & Security

### Data Protection:
- ✅ Passwords hashed (SHA-256)
- ✅ No plain text passwords stored
- ✅ User data isolated
- ✅ Activity logging for audit
- ✅ Access controls (admin only)

### Compliance:
- User data stored locally (not cloud)
- Can be exported on request
- Can be deleted on request
- Activity logs for accountability

### GDPR/Privacy Considerations:
- Users should be informed of data collection
- Provide data export capabilities
- Allow account deletion
- Implement data retention policies

---

## 🛠️ Database Management

### Backup Database:
```bash
# Backup
cp chatbot_data/users.db chatbot_data/users_backup_$(date +%Y%m%d).db

# Restore
cp chatbot_data/users_backup_20251106.db chatbot_data/users.db
```

### Export Data:
```bash
# Export to CSV
sqlite3 chatbot_data/users.db <<EOF
.headers on
.mode csv
.output users_export.csv
SELECT * FROM users;
.output conversations_export.csv
SELECT * FROM conversations;
EOF
```

### Database Statistics:
```bash
# Check database size
du -h chatbot_data/users.db

# Table row counts
sqlite3 chatbot_data/users.db "SELECT 
    'users' as table_name, COUNT(*) as rows FROM users
    UNION ALL SELECT 'conversations', COUNT(*) FROM conversations
    UNION ALL SELECT 'login_history', COUNT(*) FROM login_history
    UNION ALL SELECT 'usage_analytics', COUNT(*) FROM usage_analytics
    UNION ALL SELECT 'model_stats', COUNT(*) FROM model_stats
    UNION ALL SELECT 'activity_log', COUNT(*) FROM activity_log;"
```

---

## 📋 Sample Queries

### Top 10 Most Active Users:
```sql
SELECT 
    u.username,
    u.full_name,
    u.email,
    u.total_logins,
    COUNT(c.id) as total_messages,
    SUM(c.tokens_generated) as total_tokens
FROM users u
LEFT JOIN conversations c ON u.username = c.username
GROUP BY u.username
ORDER BY total_messages DESC
LIMIT 10;
```

### Recent Activity (Last Hour):
```sql
SELECT 
    username,
    event_type,
    details,
    timestamp
FROM activity_log
WHERE timestamp >= datetime('now', '-1 hour')
ORDER BY timestamp DESC;
```

### Model Performance Comparison:
```sql
SELECT 
    model_used,
    COUNT(*) as uses,
    AVG(response_time_ms) as avg_time_ms,
    AVG(tokens_generated) as avg_tokens,
    MIN(response_time_ms) as fastest,
    MAX(response_time_ms) as slowest
FROM conversations
WHERE model_used IS NOT NULL
GROUP BY model_used;
```

### User Retention Analysis:
```sql
SELECT 
    DATE(created_at) as signup_date,
    COUNT(*) as signups,
    SUM(CASE WHEN total_logins > 1 THEN 1 ELSE 0 END) as returned_users,
    ROUND(100.0 * SUM(CASE WHEN total_logins > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as retention_rate
FROM users
GROUP BY signup_date
ORDER BY signup_date DESC;
```

---

## 💡 Use Cases

### For Educators:
- Track student engagement
- Identify students needing help
- Measure learning outcomes
- Optimize teaching materials

### For Administrators:
- Monitor system usage
- Plan resource allocation
- Identify technical issues
- Security monitoring

### For Researchers:
- User behavior analysis
- AI interaction patterns
- Model performance studies
- Educational effectiveness

---

## ✅ Data Collection Summary

**What's Collected:**
1. User profiles (name, email, role)
2. Login activity (all attempts, timestamps)
3. Conversation content (messages, responses)
4. Performance metrics (tokens, response times)
5. Model usage (which models, how often)
6. System events (complete audit trail)

**How It's Used:**
- Admin dashboard displays
- Analytics reports
- Performance monitoring
- User management
- Security auditing

**Privacy:**
- All data stored locally
- No external sharing
- Admin-only access
- Secure hashing
- Audit trail

---

**Created:** November 6, 2025  
**Database:** SQLite with 6 tables  
**Status:** ✅ Comprehensive data collection active  
**Admin Access:** Login with admin/admin to view all data

